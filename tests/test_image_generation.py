"""
Tests for image generation using LiteLLM proxy (via direct httpx calls).

These tests verify image generation capabilities through the LiteLLM proxy.
Generated images are saved to tests/outputs/images/ for inspection.
"""
import base64
import json
from datetime import datetime
from pathlib import Path

import httpx
import pytest

# Import our proxy client instead of litellm SDK
from backend.litellm_proxy_client import (
    image_generations,
    extract_image_bytes,
    ProxyClientError,
)


class TestImageGeneration:
    """Test suite for LiteLLM image generation."""

    @pytest.fixture(autouse=True)
    def _check_image_model(self, litellm_config):
        """Skip tests if IMAGE_GEN_MODEL is not configured."""
        if not litellm_config.get("image_model"):
            pytest.skip("IMAGE_GEN_MODEL environment variable not set")

    async def _save_image_from_response(self, response: dict, output_path: Path, metadata: dict) -> Path:
        """Helper to save image from proxy response dict."""
        # Extract image bytes using our helper
        image_bytes, url = await extract_image_bytes(response)
        
        image_path = output_path.with_suffix('.png')
        image_path.write_bytes(image_bytes)
        
        # Save metadata
        metadata_path = output_path.with_suffix('.json')
        if url:
            metadata["source_url"] = url
        metadata_path.write_text(json.dumps(metadata, indent=2))
        
        return image_path

    @pytest.mark.asyncio
    async def test_simple_image_generation(self, litellm_config, image_outputs_dir):
        """Test basic image generation with a simple prompt."""
        prompt = "A beautiful sunset over the ocean with orange and purple clouds"
        
        response = await image_generations(
            model=litellm_config["image_model"],
            prompt=prompt,
            base_url=litellm_config["base_url"],
            api_key=litellm_config["api_key"],
            size="1024x1024",
        )

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_path = image_outputs_dir / f"simple_image_{timestamp}"
        
        image_path = await self._save_image_from_response(
            response, 
            output_path,
            {
                "model": litellm_config["image_model"],
                "prompt": prompt,
                "size": "1024x1024",
            }
        )

        assert image_path.exists()
        assert image_path.stat().st_size > 1000  # Should be more than 1KB
        print(f"\n✓ Image generated and saved to: {image_path}")

    @pytest.mark.asyncio
    async def test_food_image_generation(self, litellm_config, image_outputs_dir):
        """Test food image generation (similar to menu app usage)."""
        item_name = "Margherita Pizza"
        description = "Classic Italian pizza with fresh tomatoes, mozzarella, and basil"
        
        prompt = f"Create a high-quality, appetizing photo of this menu item: {item_name}. Description: {description}"

        response = await image_generations(
            model=litellm_config["image_model"],
            prompt=prompt,
            base_url=litellm_config["base_url"],
            api_key=litellm_config["api_key"],
            size="1024x1024",
        )

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_path = image_outputs_dir / f"food_{item_name.lower().replace(' ', '_')}_{timestamp}"
        
        image_path = await self._save_image_from_response(
            response,
            output_path,
            {
                "model": litellm_config["image_model"],
                "item_name": item_name,
                "description": description,
                "prompt": prompt,
            }
        )

        assert image_path.exists()
        print(f"\n✓ Food image for '{item_name}' saved to: {image_path}")

    @pytest.mark.asyncio
    async def test_multiple_food_items(self, litellm_config, image_outputs_dir):
        """Test generating images for multiple menu items."""
        menu_items = [
            {"name": "Caesar Salad", "description": "Fresh romaine lettuce with parmesan and croutons"},
            {"name": "Chocolate Lava Cake", "description": "Warm chocolate cake with molten center"},
            {"name": "Grilled Salmon", "description": "Atlantic salmon with lemon butter sauce"},
        ]

        generated_images = []
        
        for item in menu_items:
            prompt = f"Create a high-quality, appetizing photo of: {item['name']}. {item['description']}"
            
            try:
                response = await image_generations(
                    model=litellm_config["image_model"],
                    prompt=prompt,
                    base_url=litellm_config["base_url"],
                    api_key=litellm_config["api_key"],
                    size="1024x1024",
                )

                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                safe_name = item["name"].lower().replace(' ', '_')
                output_path = image_outputs_dir / f"menu_{safe_name}_{timestamp}"
                
                image_path = await self._save_image_from_response(
                    response,
                    output_path,
                    {
                        "model": litellm_config["image_model"],
                        "item": item,
                        "prompt": prompt,
                    }
                )
                
                generated_images.append({"item": item["name"], "path": str(image_path)})
                print(f"  ✓ Generated: {item['name']}")
                
            except Exception as e:
                print(f"  ✗ Failed for {item['name']}: {e}")
                generated_images.append({"item": item["name"], "error": str(e)})

        # Save summary
        summary_path = image_outputs_dir / f"batch_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        summary_path.write_text(json.dumps({
            "model": litellm_config["image_model"],
            "items_requested": len(menu_items),
            "results": generated_images,
        }, indent=2))

        successful = [img for img in generated_images if "path" in img]
        assert len(successful) > 0, "At least one image should be generated"
        print(f"\n✓ Generated {len(successful)}/{len(menu_items)} images")
        print(f"✓ Summary saved to: {summary_path}")

    @pytest.mark.asyncio
    async def test_artistic_style_image(self, litellm_config, image_outputs_dir):
        """Test image generation with artistic style specification."""
        prompt = (
            "A steaming bowl of ramen in Japanese anime art style, "
            "with vibrant colors, steam rising, chopsticks resting on the bowl"
        )

        response = await image_generations(
            model=litellm_config["image_model"],
            prompt=prompt,
            base_url=litellm_config["base_url"],
            api_key=litellm_config["api_key"],
            size="1024x1024",
        )

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_path = image_outputs_dir / f"artistic_ramen_{timestamp}"
        
        image_path = await self._save_image_from_response(
            response,
            output_path,
            {
                "model": litellm_config["image_model"],
                "prompt": prompt,
                "style": "anime",
            }
        )

        assert image_path.exists()
        print(f"\n✓ Artistic image saved to: {image_path}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
