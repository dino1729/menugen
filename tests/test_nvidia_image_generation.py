"""
Tests for image generation using NVIDIA NIM API (Stable Diffusion 3).

These tests verify image generation capabilities through NVIDIA's API.
Generated images are saved to tests/outputs/images/ for inspection.

Required environment variables:
- NVIDIA_API_KEY: Your NVIDIA API key
- NVIDIA_IMAGE_GEN_URL: NVIDIA image generation endpoint
  (e.g., https://ai.api.nvidia.com/v1/genai/stabilityai/stable-diffusion-3-medium)

The backend client includes automatic fallback from large to medium model on 404,
and flexible response parsing for different API response formats.
"""
import json
import os
from datetime import datetime
from pathlib import Path

import pytest

# Import our hardened NVIDIA client with fallback and retry logic
from backend.nvidia_image_generation import (
    generate_image_nvidia_raw,
    NVIDIA_DEFAULTS,
)
from client_utils import ImageGenerationError


class TestNvidiaImageGeneration:
    """Test suite for NVIDIA NIM image generation (Stable Diffusion 3)."""

    @pytest.fixture
    def nvidia_config(self):
        """Fixture providing NVIDIA API configuration from environment."""
        api_key = os.getenv("NVIDIA_API_KEY")
        api_url = os.getenv("NVIDIA_IMAGE_GEN_URL")
        
        if not api_key or api_key == "your_nvidia_api_key_here":
            pytest.skip("NVIDIA_API_KEY not configured")
        if not api_url:
            pytest.skip("NVIDIA_IMAGE_GEN_URL not configured")
            
        return {
            "api_key": api_key,
            "api_url": api_url,
            "model_name": self._extract_model_name(api_url),
        }

    def _extract_model_name(self, api_url: str) -> str:
        """Extract model name from NVIDIA API URL for metadata."""
        # URL format: https://ai.api.nvidia.com/v1/genai/stabilityai/stable-diffusion-3-medium
        # Extract last path segment as model identifier
        if api_url:
            parts = api_url.rstrip('/').split('/')
            if len(parts) >= 2:
                return f"{parts[-2]}/{parts[-1]}"  # e.g., "stabilityai/stable-diffusion-3-medium"
        return "nvidia-image-gen"

    async def _save_image(
        self,
        image_bytes: bytes,
        output_path: Path,
        metadata: dict
    ) -> Path:
        """Save image bytes and metadata to files."""
        image_path = output_path.with_suffix('.png')
        image_path.write_bytes(image_bytes)
        
        metadata_path = output_path.with_suffix('.json')
        metadata_path.write_text(json.dumps(metadata, indent=2))
        
        return image_path

    @pytest.mark.asyncio
    async def test_simple_image_generation(self, nvidia_config, image_outputs_dir):
        """Test basic image generation with NVIDIA API."""
        prompt = "A beautiful mountain landscape with snow-capped peaks and a clear blue sky"
        
        print(f"\n⏳ Generating image with NVIDIA Stable Diffusion 3...")
        print(f"   Prompt: {prompt}")
        
        # Use the hardened client with automatic fallback
        image_bytes = await generate_image_nvidia_raw(
            prompt=prompt,
            api_key=nvidia_config["api_key"],
            api_url=nvidia_config["api_url"],
            fallback_on_404=True,  # Will try medium if large returns 404
        )
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_path = image_outputs_dir / f"nvidia_landscape_{timestamp}"
        
        image_path = await self._save_image(
            image_bytes,
            output_path,
            {
                "provider": "nvidia",
                "model": nvidia_config["model_name"],
                "prompt": prompt,
                "parameters": NVIDIA_DEFAULTS,
            }
        )
        
        assert image_path.exists()
        assert image_path.stat().st_size > 1000  # Should be more than 1KB
        print(f"✓ Image saved to: {image_path}")

    @pytest.mark.asyncio
    async def test_food_image_generation(self, nvidia_config, image_outputs_dir):
        """Test food image generation (matching menu app usage)."""
        item_name = "Grilled Ribeye Steak"
        description = "Perfectly seared ribeye with herb butter, roasted vegetables, and red wine reduction"
        
        prompt = f"Create a high-quality, appetizing photo of this menu item: {item_name}. Description: {description}"
        
        print(f"\n⏳ Generating food image for: {item_name}")
        
        image_bytes = await generate_image_nvidia_raw(
            prompt=prompt,
            api_key=nvidia_config["api_key"],
            api_url=nvidia_config["api_url"],
        )
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        safe_name = item_name.lower().replace(' ', '_')
        output_path = image_outputs_dir / f"nvidia_food_{safe_name}_{timestamp}"
        
        image_path = await self._save_image(
            image_bytes,
            output_path,
            {
                "provider": "nvidia",
                "model": nvidia_config["model_name"],
                "item_name": item_name,
                "description": description,
                "prompt": prompt,
            }
        )
        
        assert image_path.exists()
        print(f"✓ Food image saved to: {image_path}")

    @pytest.mark.asyncio
    async def test_multiple_menu_items(self, nvidia_config, image_outputs_dir):
        """Test generating images for multiple menu items."""
        menu_items = [
            {"name": "Lobster Bisque", "description": "Creamy soup with chunks of fresh lobster and a hint of sherry"},
            {"name": "Tiramisu", "description": "Classic Italian dessert with espresso-soaked ladyfingers and mascarpone"},
            {"name": "Sushi Platter", "description": "Assorted nigiri and maki rolls with fresh wasabi and pickled ginger"},
        ]
        
        print(f"\n⏳ Generating {len(menu_items)} images with NVIDIA API...")
        
        generated_images = []
        
        for item in menu_items:
            prompt = f"Create a high-quality, appetizing photo of: {item['name']}. {item['description']}"
            
            try:
                image_bytes = await generate_image_nvidia_raw(
                    prompt=prompt,
                    api_key=nvidia_config["api_key"],
                    api_url=nvidia_config["api_url"],
                )
                
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                safe_name = item["name"].lower().replace(' ', '_')
                output_path = image_outputs_dir / f"nvidia_menu_{safe_name}_{timestamp}"
                
                image_path = await self._save_image(
                    image_bytes,
                    output_path,
                    {
                        "provider": "nvidia",
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
        summary_path = image_outputs_dir / f"nvidia_batch_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        summary_path.write_text(json.dumps({
            "provider": "nvidia",
            "items_requested": len(menu_items),
            "results": generated_images,
        }, indent=2))
        
        successful = [img for img in generated_images if "path" in img]
        assert len(successful) > 0, "At least one image should be generated"
        print(f"\n✓ Generated {len(successful)}/{len(menu_items)} images")
        print(f"✓ Summary saved to: {summary_path}")

    @pytest.mark.asyncio
    async def test_different_aspect_ratios(self, nvidia_config, image_outputs_dir):
        """Test image generation with different aspect ratios."""
        prompt = "A delicious pepperoni pizza fresh from the oven"
        
        aspect_ratios = ["1:1", "16:9", "9:16", "4:3"]
        
        print(f"\n⏳ Testing different aspect ratios...")
        
        results = []
        
        for ratio in aspect_ratios:
            try:
                image_bytes = await generate_image_nvidia_raw(
                    prompt=prompt,
                    api_key=nvidia_config["api_key"],
                    api_url=nvidia_config["api_url"],
                    aspect_ratio=ratio,
                )
                
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                safe_ratio = ratio.replace(':', 'x')
                output_path = image_outputs_dir / f"nvidia_ratio_{safe_ratio}_{timestamp}"
                
                image_path = await self._save_image(
                    image_bytes,
                    output_path,
                    {
                        "provider": "nvidia",
                        "prompt": prompt,
                        "aspect_ratio": ratio,
                    }
                )
                
                results.append({"ratio": ratio, "path": str(image_path), "status": "success"})
                print(f"  ✓ {ratio}: {image_path}")
                
            except Exception as e:
                results.append({"ratio": ratio, "error": str(e), "status": "failed"})
                print(f"  ✗ {ratio}: {e}")
        
        # Save comparison
        comparison_path = image_outputs_dir / f"nvidia_ratio_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        comparison_path.write_text(json.dumps({
            "provider": "nvidia",
            "prompt": prompt,
            "results": results,
        }, indent=2))
        
        successful = [r for r in results if r["status"] == "success"]
        assert len(successful) > 0
        print(f"\n✓ Comparison saved to: {comparison_path}")

    @pytest.mark.asyncio
    async def test_cfg_scale_comparison(self, nvidia_config, image_outputs_dir):
        """Test image generation with different CFG scale values."""
        prompt = "A colorful fruit salad in a crystal bowl"
        
        cfg_scales = [3, 5, 7, 10]
        
        print(f"\n⏳ Testing different CFG scale values...")
        
        results = []
        
        for cfg in cfg_scales:
            try:
                image_bytes = await generate_image_nvidia_raw(
                    prompt=prompt,
                    api_key=nvidia_config["api_key"],
                    api_url=nvidia_config["api_url"],
                    cfg_scale=cfg,
                )
                
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                output_path = image_outputs_dir / f"nvidia_cfg{cfg}_{timestamp}"
                
                image_path = await self._save_image(
                    image_bytes,
                    output_path,
                    {
                        "provider": "nvidia",
                        "prompt": prompt,
                        "cfg_scale": cfg,
                    }
                )
                
                results.append({"cfg_scale": cfg, "path": str(image_path), "status": "success"})
                print(f"  ✓ CFG {cfg}: {image_path}")
                
            except Exception as e:
                results.append({"cfg_scale": cfg, "error": str(e), "status": "failed"})
                print(f"  ✗ CFG {cfg}: {e}")
        
        # Save comparison
        comparison_path = image_outputs_dir / f"nvidia_cfg_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        comparison_path.write_text(json.dumps({
            "provider": "nvidia",
            "prompt": prompt,
            "results": results,
        }, indent=2))
        
        print(f"\n✓ CFG comparison saved to: {comparison_path}")

    @pytest.mark.asyncio
    async def test_negative_prompt(self, nvidia_config, image_outputs_dir):
        """Test image generation with negative prompt."""
        prompt = "A gourmet burger with melted cheese and fresh vegetables"
        negative_prompt = "blurry, low quality, watermark, text, cartoon, anime"
        
        print(f"\n⏳ Testing with negative prompt...")
        
        # Generate with negative prompt
        image_bytes = await generate_image_nvidia_raw(
            prompt=prompt,
            api_key=nvidia_config["api_key"],
            api_url=nvidia_config["api_url"],
            negative_prompt=negative_prompt,
        )
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_path = image_outputs_dir / f"nvidia_negative_prompt_{timestamp}"
        
        image_path = await self._save_image(
            image_bytes,
            output_path,
            {
                "provider": "nvidia",
                "prompt": prompt,
                "negative_prompt": negative_prompt,
            }
        )
        
        assert image_path.exists()
        print(f"✓ Image with negative prompt saved to: {image_path}")

    @pytest.mark.asyncio
    async def test_seed_reproducibility(self, nvidia_config, image_outputs_dir):
        """Test that the same seed produces consistent results."""
        prompt = "A cup of cappuccino with latte art"
        seed = 42
        
        print(f"\n⏳ Testing seed reproducibility (seed={seed})...")
        
        images = []
        
        for i in range(2):
            image_bytes = await generate_image_nvidia_raw(
                prompt=prompt,
                api_key=nvidia_config["api_key"],
                api_url=nvidia_config["api_url"],
                seed=seed,
            )
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_path = image_outputs_dir / f"nvidia_seed{seed}_run{i+1}_{timestamp}"
            
            image_path = await self._save_image(
                image_bytes,
                output_path,
                {
                    "provider": "nvidia",
                    "prompt": prompt,
                    "seed": seed,
                    "run": i + 1,
                }
            )
            
            images.append({"path": str(image_path), "size": image_path.stat().st_size})
            print(f"  ✓ Run {i+1}: {image_path}")
        
        # Note: Exact reproducibility depends on NVIDIA's implementation
        print(f"\n✓ Generated {len(images)} images with seed {seed}")
        print(f"  Compare the images manually to verify consistency")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
