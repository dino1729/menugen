"""
End-to-End Test for MenuGen Application Flow.

This test simulates the complete MenuGen pipeline:
1. Start with menu items (or parse from image if provided)
2. Generate/simplify descriptions for each item using LLM
3. Generate images for each item using image generation model
4. Save all outputs (descriptions + images) to test outputs

The test can run in two modes:
- Mock mode: Uses predefined sample menu items (no vision model needed)
- Full mode: Parses a real menu image (requires vision-capable LLM)

All outputs are saved to tests/outputs/e2e/ for inspection.

Uses direct httpx calls to the LiteLLM proxy to avoid SDK bypass issues.
"""
import asyncio
import base64
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import httpx
import pytest

# Import our proxy client and NVIDIA client
from backend.litellm_proxy_client import (
    chat_completions,
    image_generations,
    extract_chat_content,
    extract_json_from_text,
    extract_image_bytes,
    ProxyClientError,
)
from backend.nvidia_image_generation import generate_image_nvidia_raw


class MenuGenE2ETest:
    """End-to-end test runner for MenuGen pipeline."""

    def __init__(
        self,
        base_url: str,
        api_key: str,
        text_model: str,
        image_model: str,
        output_dir: Path,
        vision_model: Optional[str] = None,
        image_provider: str = "litellm",
        nvidia_api_key: Optional[str] = None,
        nvidia_api_url: Optional[str] = None,
    ):
        self.base_url = base_url
        self.api_key = api_key
        self.text_model = text_model
        self.vision_model = vision_model or text_model  # Falls back to text_model
        self.image_model = image_model
        self.output_dir = output_dir
        self.image_provider = image_provider
        self.nvidia_api_key = nvidia_api_key
        self.nvidia_api_url = nvidia_api_url
        
        # Create output directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "images").mkdir(exist_ok=True)
        
        # Results tracking
        self.results = {
            "timestamp": datetime.now().isoformat(),
            "config": {
                "text_model": text_model,
                "vision_model": self.vision_model,
                "image_model": image_model,
                "image_provider": image_provider,
            },
            "items": [],
            "summary": {},
        }

    async def parse_menu_image(self, image_content: bytes) -> Dict:
        """
        Parse menu image using vision model.
        
        Returns dict with 'items' array containing menu items.
        """
        base64_image = base64.b64encode(image_content).decode('utf-8')
        image_url = f"data:image/jpeg;base64,{base64_image}"

        prompt = (
            "You are a helpful assistant that extracts structured menu data from images. "
            "Given a photo of a restaurant menu, return a JSON object with this exact structure: "
            '{"items": [{"name": "Item Name", "description": "Item description if available", "section": "Section name if available"}, ...]}. '
            "Each item in the 'items' array should be a flat object with 'name', optional 'description', and optional 'section' fields. "
            "Do NOT nest items under sections. Flatten all items into a single 'items' array. "
            "Respond ONLY with valid JSON in this exact format."
        )

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": image_url}}
                ]
            }
        ]

        response = await chat_completions(
            model=self.vision_model,
            messages=messages,
            base_url=self.base_url,
            api_key=self.api_key,
        )

        content = extract_chat_content(response)
        return extract_json_from_text(content)

    async def simplify_description(self, item: Dict) -> str:
        """
        Generate or simplify menu item description.
        
        If description exists, rephrases it in simple English.
        If no description, generates one based on the item name.
        """
        item_name = item.get('name', 'Unknown')
        description = item.get('description')
        
        if description:
            prompt = (
                f"Rephrase the following menu item description as a single, complete sentence in simple English. "
                f"Explain any potentially unfamiliar culinary terms. "
                f"Original item name: '{item_name}'. Original description: '{description}' "
                f"Rephrased sentence in simple English:"
            )
            system_message = "You rephrase menu descriptions into single, simple English sentences."
        else:
            prompt = (
                f"Generate a simple, concise, and appetizing description for the menu item named '{item_name}' "
                f"as a single, complete sentence in simple English. "
                f"Generated sentence in simple English:"
            )
            system_message = "You generate simple and appetizing menu descriptions as single, complete sentences."

        response = await chat_completions(
            model=self.text_model,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": prompt}
            ],
            base_url=self.base_url,
            api_key=self.api_key,
            max_tokens=100
        )

        return extract_chat_content(response).strip('"')

    async def generate_image_litellm(self, item: Dict) -> bytes:
        """Generate image using LiteLLM proxy."""
        item_name = item.get('name', 'Unknown')
        description = item.get('description', '')
        
        prompt = f"Create a high-quality, appetizing photo of this menu item: {item_name}. Description: {description}"

        response = await image_generations(
            model=self.image_model,
            prompt=prompt,
            base_url=self.base_url,
            api_key=self.api_key,
            size="1024x1024"
        )

        image_bytes, _ = await extract_image_bytes(response)
        return image_bytes

    async def generate_image_nvidia(self, item: Dict) -> bytes:
        """Generate image using NVIDIA API (Stable Diffusion 3)."""
        if not self.nvidia_api_key or not self.nvidia_api_url:
            raise ValueError("NVIDIA API key and URL required for NVIDIA image generation")
        
        item_name = item.get('name', 'Unknown')
        description = item.get('description', '')
        
        prompt = f"Create a high-quality, appetizing photo of this menu item: {item_name}. Description: {description}"
        
        # Use hardened NVIDIA client with fallback
        return await generate_image_nvidia_raw(
            prompt=prompt,
            api_key=self.nvidia_api_key,
            api_url=self.nvidia_api_url,
            fallback_on_404=True,
        )

    async def generate_image(self, item: Dict) -> bytes:
        """Generate image using configured provider."""
        if self.image_provider == "nvidia":
            return await self.generate_image_nvidia(item)
        else:
            return await self.generate_image_litellm(item)

    def sanitize_filename(self, name: str) -> str:
        """Create safe filename from item name."""
        import re
        name = re.sub(r'[<>:"/\\|?*]', '', name)
        name = re.sub(r'\s+', '_', name)
        return name[:50]

    async def process_item(self, item: Dict, index: int) -> Dict:
        """
        Process a single menu item through the full pipeline.
        
        Returns result dict with status, description, and image path.
        """
        item_name = item.get('name', f'Item_{index}')
        result = {
            "index": index,
            "original_item": item.copy(),
            "name": item_name,
            "status": "pending",
            "steps": {},
        }
        
        print(f"\n{'='*60}")
        print(f"Processing item {index + 1}: {item_name}")
        print(f"{'='*60}")
        
        # Step 1: Generate/simplify description
        print(f"  [1/2] Generating description...")
        try:
            description = await self.simplify_description(item)
            item['description'] = description  # Update for image generation
            result['processed_description'] = description
            result['steps']['description'] = {"status": "success", "output": description}
            print(f"        ✓ Description: {description[:80]}...")
        except Exception as e:
            result['steps']['description'] = {"status": "failed", "error": str(e)}
            print(f"        ✗ Failed: {e}")
            # Use original description or empty for image generation
            result['processed_description'] = item.get('description', '')
        
        # Step 2: Generate image
        print(f"  [2/2] Generating image ({self.image_provider})...")
        try:
            image_bytes = await self.generate_image(item)
            
            # Save image
            safe_name = self.sanitize_filename(item_name)
            timestamp = datetime.now().strftime('%H%M%S')
            image_filename = f"{index+1:02d}_{safe_name}_{timestamp}.png"
            image_path = self.output_dir / "images" / image_filename
            image_path.write_bytes(image_bytes)
            
            result['image_path'] = str(image_path)
            result['image_size_kb'] = round(len(image_bytes) / 1024, 2)
            result['steps']['image'] = {"status": "success", "path": str(image_path)}
            print(f"        ✓ Image saved: {image_filename} ({result['image_size_kb']} KB)")
        except Exception as e:
            result['steps']['image'] = {"status": "failed", "error": str(e)}
            print(f"        ✗ Failed: {e}")
        
        # Determine overall status
        desc_ok = result['steps'].get('description', {}).get('status') == 'success'
        img_ok = result['steps'].get('image', {}).get('status') == 'success'
        
        if desc_ok and img_ok:
            result['status'] = 'success'
        elif desc_ok or img_ok:
            result['status'] = 'partial'
        else:
            result['status'] = 'failed'
        
        return result

    async def run(self, menu_items: List[Dict]) -> Dict:
        """
        Run the full E2E pipeline for all menu items.
        
        Returns complete results dict.
        """
        print(f"\n{'#'*60}")
        print(f"# MenuGen E2E Test")
        print(f"# Items to process: {len(menu_items)}")
        print(f"# Text model: {self.text_model}")
        print(f"# Vision model: {self.vision_model}")
        print(f"# Image provider: {self.image_provider}")
        print(f"# Output directory: {self.output_dir}")
        print(f"{'#'*60}")
        
        self.results['input_items'] = menu_items
        
        # Process each item
        for idx, item in enumerate(menu_items):
            result = await self.process_item(item, idx)
            self.results['items'].append(result)
        
        # Generate summary
        total = len(self.results['items'])
        success = sum(1 for r in self.results['items'] if r['status'] == 'success')
        partial = sum(1 for r in self.results['items'] if r['status'] == 'partial')
        failed = sum(1 for r in self.results['items'] if r['status'] == 'failed')
        
        self.results['summary'] = {
            "total_items": total,
            "success": success,
            "partial": partial,
            "failed": failed,
            "success_rate": f"{(success/total)*100:.1f}%" if total > 0 else "N/A",
        }
        
        # Save results
        results_path = self.output_dir / "results.json"
        results_path.write_text(json.dumps(self.results, indent=2))
        
        # Print summary
        print(f"\n{'#'*60}")
        print(f"# E2E Test Complete")
        print(f"# Total: {total} | Success: {success} | Partial: {partial} | Failed: {failed}")
        print(f"# Results saved to: {results_path}")
        print(f"{'#'*60}\n")
        
        return self.results


# Sample menu items for testing without a real menu image
SAMPLE_MENU_ITEMS = [
    {
        "name": "Margherita Pizza",
        "description": "Fresh tomatoes, mozzarella, basil",
        "section": "Pizzas"
    },
    {
        "name": "Caesar Salad",
        "description": "Romaine lettuce, parmesan, croutons, caesar dressing",
        "section": "Salads"
    },
    {
        "name": "Grilled Salmon",
        "description": "Atlantic salmon with lemon butter sauce and seasonal vegetables",
        "section": "Main Courses"
    },
    {
        "name": "Tiramisu",
        "description": "Classic Italian dessert with espresso and mascarpone",
        "section": "Desserts"
    },
    {
        "name": "Truffle Risotto",
        "description": "",  # No description - will be generated
        "section": "Main Courses"
    },
]


class TestE2EFlow:
    """Pytest test class for E2E flow."""

    @pytest.fixture
    def e2e_outputs_dir(self, outputs_dir):
        """Create unique output directory for this test run."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        path = outputs_dir / "e2e" / f"run_{timestamp}"
        path.mkdir(parents=True, exist_ok=True)
        return path

    @pytest.mark.asyncio
    async def test_e2e_with_sample_items_litellm(self, litellm_config, e2e_outputs_dir):
        """
        E2E test using sample menu items with LiteLLM image generation.
        
        This test simulates the full flow without requiring a menu image.
        """
        runner = MenuGenE2ETest(
            base_url=litellm_config["base_url"],
            api_key=litellm_config["api_key"],
            text_model=litellm_config["text_model"],
            vision_model=litellm_config["vision_model"],
            image_model=litellm_config["image_model"],
            output_dir=e2e_outputs_dir / "litellm",
            image_provider="litellm",
        )
        
        results = await runner.run(SAMPLE_MENU_ITEMS[:3])  # Test with 3 items
        
        assert results['summary']['total_items'] == 3
        assert results['summary']['success'] > 0, "At least one item should complete successfully"
        
        print(f"\n✓ LiteLLM E2E test completed")
        print(f"  Results: {e2e_outputs_dir / 'litellm' / 'results.json'}")

    @pytest.mark.asyncio
    async def test_e2e_with_sample_items_nvidia(self, litellm_config, e2e_outputs_dir):
        """
        E2E test using sample menu items with NVIDIA image generation.
        
        Requires NVIDIA_API_KEY and NVIDIA_IMAGE_GEN_URL to be configured.
        """
        nvidia_api_key = os.getenv("NVIDIA_API_KEY")
        nvidia_api_url = os.getenv("NVIDIA_IMAGE_GEN_URL")
        
        if not nvidia_api_key or nvidia_api_key == "your_nvidia_api_key_here":
            pytest.skip("NVIDIA_API_KEY not configured")
        if not nvidia_api_url:
            pytest.skip("NVIDIA_IMAGE_GEN_URL not configured")
        
        # Extract model name from URL for metadata
        nvidia_model = "nvidia-image-gen"
        if nvidia_api_url:
            parts = nvidia_api_url.rstrip('/').split('/')
            if len(parts) >= 2:
                nvidia_model = f"{parts[-2]}/{parts[-1]}"
        
        runner = MenuGenE2ETest(
            base_url=litellm_config["base_url"],
            api_key=litellm_config["api_key"],
            text_model=litellm_config["text_model"],
            vision_model=litellm_config["vision_model"],
            image_model=nvidia_model,
            output_dir=e2e_outputs_dir / "nvidia",
            image_provider="nvidia",
            nvidia_api_key=nvidia_api_key,
            nvidia_api_url=nvidia_api_url,
        )
        
        results = await runner.run(SAMPLE_MENU_ITEMS[:3])  # Test with 3 items
        
        assert results['summary']['total_items'] == 3
        assert results['summary']['success'] > 0, "At least one item should complete successfully"
        
        print(f"\n✓ NVIDIA E2E test completed")
        print(f"  Results: {e2e_outputs_dir / 'nvidia' / 'results.json'}")

    @pytest.mark.asyncio
    async def test_e2e_full_menu(self, litellm_config, e2e_outputs_dir):
        """
        E2E test processing all sample menu items.
        
        This is a longer test that processes the complete sample menu.
        """
        runner = MenuGenE2ETest(
            base_url=litellm_config["base_url"],
            api_key=litellm_config["api_key"],
            text_model=litellm_config["text_model"],
            vision_model=litellm_config["vision_model"],
            image_model=litellm_config["image_model"],
            output_dir=e2e_outputs_dir / "full_menu",
            image_provider="litellm",
        )
        
        results = await runner.run(SAMPLE_MENU_ITEMS)
        
        assert results['summary']['total_items'] == len(SAMPLE_MENU_ITEMS)
        
        # Generate HTML report for easy viewing
        html_report = generate_html_report(results, e2e_outputs_dir / "full_menu")
        report_path = e2e_outputs_dir / "full_menu" / "report.html"
        report_path.write_text(html_report)
        
        print(f"\n✓ Full menu E2E test completed")
        print(f"  HTML Report: {report_path}")


def generate_html_report(results: Dict, output_dir: Path) -> str:
    """Generate an HTML report for easy visualization of results."""
    items_html = ""
    
    for item in results['items']:
        status_color = {
            'success': '#28a745',
            'partial': '#ffc107',
            'failed': '#dc3545',
        }.get(item['status'], '#6c757d')
        
        image_html = ""
        if 'image_path' in item:
            # Get relative path for HTML
            image_filename = Path(item['image_path']).name
            image_html = f'<img src="images/{image_filename}" alt="{item["name"]}" style="max-width: 300px; border-radius: 8px;">'
        else:
            image_html = '<p style="color: #999;">Image generation failed</p>'
        
        items_html += f"""
        <div style="border: 1px solid #ddd; border-radius: 8px; padding: 16px; margin: 16px 0; background: #fff;">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
                <h3 style="margin: 0;">{item['name']}</h3>
                <span style="background: {status_color}; color: white; padding: 4px 12px; border-radius: 4px; font-size: 12px;">
                    {item['status'].upper()}
                </span>
            </div>
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 16px;">
                <div>
                    <h4 style="margin: 0 0 8px 0; color: #666;">Original</h4>
                    <p style="margin: 0; color: #333;">{item['original_item'].get('description', 'No description')}</p>
                    <h4 style="margin: 16px 0 8px 0; color: #666;">Processed</h4>
                    <p style="margin: 0; color: #333;">{item.get('processed_description', 'N/A')}</p>
                </div>
                <div style="text-align: center;">
                    {image_html}
                </div>
            </div>
        </div>
        """
    
    summary = results['summary']
    
    return f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>MenuGen E2E Test Report</title>
        <style>
            body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #f5f5f5; margin: 0; padding: 20px; }}
            .container {{ max-width: 900px; margin: 0 auto; }}
            .header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 24px; border-radius: 12px; margin-bottom: 24px; }}
            .summary {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; margin-bottom: 24px; }}
            .stat {{ background: white; padding: 16px; border-radius: 8px; text-align: center; }}
            .stat-value {{ font-size: 24px; font-weight: bold; color: #333; }}
            .stat-label {{ font-size: 12px; color: #666; margin-top: 4px; }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1 style="margin: 0;">MenuGen E2E Test Report</h1>
                <p style="margin: 8px 0 0 0; opacity: 0.9;">Generated: {results['timestamp']}</p>
            </div>
            
            <div class="summary">
                <div class="stat">
                    <div class="stat-value">{summary['total_items']}</div>
                    <div class="stat-label">Total Items</div>
                </div>
                <div class="stat">
                    <div class="stat-value" style="color: #28a745;">{summary['success']}</div>
                    <div class="stat-label">Success</div>
                </div>
                <div class="stat">
                    <div class="stat-value" style="color: #ffc107;">{summary['partial']}</div>
                    <div class="stat-label">Partial</div>
                </div>
                <div class="stat">
                    <div class="stat-value" style="color: #dc3545;">{summary['failed']}</div>
                    <div class="stat-label">Failed</div>
                </div>
            </div>
            
            <h2>Processed Items</h2>
            {items_html}
            
            <div style="margin-top: 24px; padding: 16px; background: #fff; border-radius: 8px;">
                <h4 style="margin: 0 0 8px 0;">Configuration</h4>
                <pre style="margin: 0; font-size: 12px; color: #666;">{json.dumps(results['config'], indent=2)}</pre>
            </div>
        </div>
    </body>
    </html>
    """


# Standalone script runner
async def run_standalone():
    """Run E2E test as standalone script."""
    from dotenv import load_dotenv
    
    # Load environment
    project_root = Path(__file__).parent.parent
    load_dotenv(project_root / ".env")
    
    # Configuration - all model names from environment (no hardcoded defaults)
    text_model = os.getenv("LLM_MODEL")
    if not text_model:
        print("ERROR: LLM_MODEL environment variable not set")
        print("Please set required environment variables in .env file")
        return
    
    config = {
        "base_url": os.getenv("LITELLM_BASE_URL", "http://localhost:4000"),
        "api_key": os.getenv("LITELLM_API_KEY"),
        "text_model": text_model,
        "vision_model": os.getenv("VISION_MODEL") or text_model,
        "image_model": os.getenv("IMAGE_GEN_MODEL"),
    }
    
    # Output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = project_root / "tests" / "outputs" / "e2e" / f"standalone_{timestamp}"
    
    # Run test
    runner = MenuGenE2ETest(
        base_url=config["base_url"],
        api_key=config["api_key"],
        text_model=config["text_model"],
        vision_model=config["vision_model"],
        image_model=config["image_model"],
        output_dir=output_dir,
        image_provider="litellm",
    )
    
    results = await runner.run(SAMPLE_MENU_ITEMS)
    
    # Generate HTML report
    html_report = generate_html_report(results, output_dir)
    report_path = output_dir / "report.html"
    report_path.write_text(html_report)
    
    print(f"\n📊 HTML Report: {report_path}")
    print(f"📁 All outputs: {output_dir}")


if __name__ == "__main__":
    # Run as standalone script
    asyncio.run(run_standalone())
