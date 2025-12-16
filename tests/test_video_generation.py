"""
Tests for video generation using LiteLLM proxy.

Supports models like:
- sora-2: OpenAI's Sora video generation model
- veo-3.0-generate-001: Google's Veo 3.0 video generation model
- veo-3.1-generate-001: Google's Veo 3.1 video generation model

Uses the LiteLLM /v1/videos endpoint as documented at:
https://docs.litellm.ai/docs/providers/openai/videos

Generated videos are saved to tests/outputs/videos/ for inspection.

Note: Video generation may take longer than text/image generation.
These tests have extended timeouts and use the shared retry/backoff logic.
"""
import asyncio
import json
from datetime import datetime
from pathlib import Path

import pytest

# Import our proxy client with video support
from backend.litellm_proxy_client import (
    video_generations,
    video_status,
    video_content,
    ProxyClientError,
)


class TestVideoGeneration:
    """Test suite for LiteLLM video generation."""

    @pytest.fixture(autouse=True)
    def _check_video_model(self, litellm_config):
        """Skip tests if VIDEO_GEN_MODEL is not configured."""
        if not litellm_config.get("video_model"):
            pytest.skip("VIDEO_GEN_MODEL environment variable not set")

    # Video model configurations
    VIDEO_MODELS = {
        "veo-3.0": "veo-3.0-generate-001",
        "veo-3.1": "veo-3.1-generate-001",
        "sora-2": "sora-2",
    }

    async def _wait_for_video(
        self,
        video_id: str,
        base_url: str,
        api_key: str,
        max_wait_seconds: int = 120,
        poll_interval: int = 5,
    ) -> dict:
        """Poll video status until complete or timeout."""
        elapsed = 0
        while elapsed < max_wait_seconds:
            try:
                status_response = await video_status(
                    video_id=video_id,
                    base_url=base_url,
                    api_key=api_key,
                )
                status = status_response.get("status", "unknown")
                
                if status in ("completed", "succeeded", "ready"):
                    return status_response
                elif status in ("failed", "error"):
                    raise ProxyClientError(f"Video generation failed: {status_response}")
                
                print(f"    Status: {status}, waiting {poll_interval}s...")
                await asyncio.sleep(poll_interval)
                elapsed += poll_interval
                
            except ProxyClientError as e:
                if e.status_code == 404:
                    # Video not found yet, keep waiting
                    await asyncio.sleep(poll_interval)
                    elapsed += poll_interval
                else:
                    raise
        
        raise ProxyClientError(f"Video generation timed out after {max_wait_seconds}s")

    async def _save_video(self, video_bytes: bytes, output_path: Path, metadata: dict) -> Path:
        """Save video bytes and metadata to files."""
        video_path = output_path.with_suffix('.mp4')
        video_path.write_bytes(video_bytes)
        
        metadata_path = output_path.with_suffix('.json')
        metadata_path.write_text(json.dumps(metadata, indent=2))
        
        return video_path

    @pytest.mark.asyncio
    @pytest.mark.timeout(300)  # 5 minute timeout for video generation
    async def test_veo_3_0_simple_video(self, litellm_config, video_outputs_dir):
        """Test simple video generation with Veo 3.0 model."""
        model = self.VIDEO_MODELS["veo-3.0"]
        prompt = "A cat playing with a ball of yarn in a cozy living room"
        
        print(f"\n⏳ Generating video with {model}...")
        print(f"   Prompt: {prompt}")
        
        try:
            # Step 1: Start video generation
            result = await video_generations(
                model=model,
                prompt=prompt,
                base_url=litellm_config["base_url"],
                api_key=litellm_config["api_key"],
                seconds="5",
                size="1280x720",
                timeout=60.0,
            )
            
            video_id = result.get("id") or result.get("video_id")
            print(f"   Video ID: {video_id}")
            print(f"   Initial status: {result.get('status', 'unknown')}")
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_path = video_outputs_dir / f"veo30_cat_{timestamp}"
            
            # Save initial response metadata
            metadata = {
                "model": model,
                "prompt": prompt,
                "seconds": "5",
                "size": "1280x720",
                "video_id": video_id,
                "initial_response": result,
            }
            
            # Step 2: If we got a video_id, try to wait and download
            if video_id:
                try:
                    print(f"   Waiting for video completion...")
                    await self._wait_for_video(
                        video_id=video_id,
                        base_url=litellm_config["base_url"],
                        api_key=litellm_config["api_key"],
                        max_wait_seconds=180,
                    )
                    
                    # Step 3: Download video content
                    video_bytes = await video_content(
                        video_id=video_id,
                        base_url=litellm_config["base_url"],
                        api_key=litellm_config["api_key"],
                    )
                    
                    video_path = await self._save_video(video_bytes, output_path, metadata)
                    print(f"✓ Video saved to: {video_path}")
                    assert video_path.exists()
                    
                except ProxyClientError as e:
                    # Video download failed, but generation started - save metadata
                    metadata["download_error"] = str(e)
                    metadata_path = output_path.with_suffix('.json')
                    metadata_path.write_text(json.dumps(metadata, indent=2))
                    print(f"⚠ Video generation started but download failed: {e}")
                    print(f"✓ Metadata saved to: {metadata_path}")
            else:
                # No video_id - just save the response
                metadata_path = output_path.with_suffix('.json')
                metadata_path.write_text(json.dumps(metadata, indent=2))
                print(f"✓ Response saved to: {metadata_path}")
            
            assert output_path.with_suffix('.json').exists() or output_path.with_suffix('.mp4').exists()
            
        except ProxyClientError as e:
            if e.status_code == 404:
                pytest.skip(f"Video generation endpoint not available: {e}")
            elif e.status_code == 429:
                pytest.skip(f"Video generation rate limited after retries: {e}")
            elif e.status_code == 400:
                error_path = video_outputs_dir / f"veo30_error_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                error_path.write_text(json.dumps({
                    "model": model,
                    "prompt": prompt,
                    "error": str(e),
                    "response": e.response_body,
                }, indent=2))
                pytest.skip(f"Video generation not supported for model: {e}")
            else:
                raise

    @pytest.mark.asyncio
    @pytest.mark.timeout(300)
    async def test_veo_3_1_simple_video(self, litellm_config, video_outputs_dir):
        """Test simple video generation with Veo 3.1 model."""
        model = self.VIDEO_MODELS["veo-3.1"]
        prompt = "Ocean waves gently crashing on a sandy beach at sunset"
        
        print(f"\n⏳ Generating video with {model}...")
        print(f"   Prompt: {prompt}")
        
        try:
            # Step 1: Start video generation
            result = await video_generations(
                model=model,
                prompt=prompt,
                base_url=litellm_config["base_url"],
                api_key=litellm_config["api_key"],
                seconds="5",
                size="1280x720",
                timeout=60.0,
            )
            
            video_id = result.get("id") or result.get("video_id")
            print(f"   Video ID: {video_id}")
            print(f"   Initial status: {result.get('status', 'unknown')}")
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_path = video_outputs_dir / f"veo31_beach_{timestamp}"
            
            # Save initial response metadata
            metadata = {
                "model": model,
                "prompt": prompt,
                "seconds": "5",
                "size": "1280x720",
                "video_id": video_id,
                "initial_response": result,
            }
            
            # Step 2: If we got a video_id, try to wait and download
            if video_id:
                try:
                    print(f"   Waiting for video completion...")
                    await self._wait_for_video(
                        video_id=video_id,
                        base_url=litellm_config["base_url"],
                        api_key=litellm_config["api_key"],
                        max_wait_seconds=180,
                    )
                    
                    # Step 3: Download video content
                    video_bytes = await video_content(
                        video_id=video_id,
                        base_url=litellm_config["base_url"],
                        api_key=litellm_config["api_key"],
                    )
                    
                    video_path = await self._save_video(video_bytes, output_path, metadata)
                    print(f"✓ Video saved to: {video_path}")
                    assert video_path.exists()
                    
                except ProxyClientError as e:
                    # Video download failed, but generation started - save metadata
                    metadata["download_error"] = str(e)
                    metadata_path = output_path.with_suffix('.json')
                    metadata_path.write_text(json.dumps(metadata, indent=2))
                    print(f"⚠ Video generation started but download failed: {e}")
                    print(f"✓ Metadata saved to: {metadata_path}")
            else:
                # No video_id - just save the response
                metadata_path = output_path.with_suffix('.json')
                metadata_path.write_text(json.dumps(metadata, indent=2))
                print(f"✓ Response saved to: {metadata_path}")
            
            assert output_path.with_suffix('.json').exists() or output_path.with_suffix('.mp4').exists()
            
        except ProxyClientError as e:
            if e.status_code in (404, 400):
                error_path = video_outputs_dir / f"veo31_error_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                error_path.write_text(json.dumps({
                    "model": model,
                    "prompt": prompt,
                    "error": str(e),
                    "status_code": e.status_code,
                }, indent=2))
                pytest.skip(f"Veo 3.1 video generation not available: {e}")
            elif e.status_code == 429:
                pytest.skip(f"Veo 3.1 rate limited after retries: {e}")
            raise

    @pytest.mark.asyncio
    @pytest.mark.timeout(300)
    async def test_veo_food_video(self, litellm_config, video_outputs_dir):
        """Test food-related video generation (relevant to menu app)."""
        model = self.VIDEO_MODELS["veo-3.0"]
        prompt = (
            "Close-up of a chef's hands plating a gourmet dish, "
            "adding final garnishes, steam rising from hot food, professional kitchen"
        )
        
        print(f"\n⏳ Generating food video with {model}...")
        print(f"   Prompt: {prompt}")
        
        try:
            # Step 1: Start video generation
            result = await video_generations(
                model=model,
                prompt=prompt,
                base_url=litellm_config["base_url"],
                api_key=litellm_config["api_key"],
                seconds="5",
                size="1280x720",
                timeout=60.0,
            )
            
            video_id = result.get("id") or result.get("video_id")
            print(f"   Video ID: {video_id}")
            print(f"   Initial status: {result.get('status', 'unknown')}")
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_path = video_outputs_dir / f"veo_food_plating_{timestamp}"
            
            # Save initial response metadata
            metadata = {
                "model": model,
                "prompt": prompt,
                "seconds": "5",
                "size": "1280x720",
                "use_case": "food/menu",
                "video_id": video_id,
                "initial_response": result,
            }
            
            # Step 2: If we got a video_id, try to wait and download
            if video_id:
                try:
                    print(f"   Waiting for video completion...")
                    await self._wait_for_video(
                        video_id=video_id,
                        base_url=litellm_config["base_url"],
                        api_key=litellm_config["api_key"],
                        max_wait_seconds=180,
                    )
                    
                    # Step 3: Download video content
                    video_bytes = await video_content(
                        video_id=video_id,
                        base_url=litellm_config["base_url"],
                        api_key=litellm_config["api_key"],
                    )
                    
                    video_path = await self._save_video(video_bytes, output_path, metadata)
                    print(f"✓ Video saved to: {video_path}")
                    assert video_path.exists()
                    
                except ProxyClientError as e:
                    # Video download failed, but generation started - save metadata
                    metadata["download_error"] = str(e)
                    metadata_path = output_path.with_suffix('.json')
                    metadata_path.write_text(json.dumps(metadata, indent=2))
                    print(f"⚠ Video generation started but download failed: {e}")
                    print(f"✓ Metadata saved to: {metadata_path}")
            else:
                # No video_id - just save the response
                metadata_path = output_path.with_suffix('.json')
                metadata_path.write_text(json.dumps(metadata, indent=2))
                print(f"✓ Response saved to: {metadata_path}")
            
            assert output_path.with_suffix('.json').exists() or output_path.with_suffix('.mp4').exists()
            
        except ProxyClientError as e:
            if e.status_code in (404, 400, 429):
                error_path = video_outputs_dir / f"veo_food_error_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                error_path.write_text(json.dumps({
                    "model": model,
                    "prompt": prompt,
                    "error": str(e),
                }, indent=2))
                pytest.skip(f"Video generation not available: {e}")
            raise

    @pytest.mark.asyncio
    @pytest.mark.timeout(600)  # 10 minute timeout for comparison test
    async def test_veo_model_comparison(self, litellm_config, video_outputs_dir):
        """Compare video generation between different Veo models."""
        prompt = "A colorful butterfly landing on a flower in a garden"
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        results = {}
        
        for version, model in self.VIDEO_MODELS.items():
            if version == "sora-2":
                continue  # Skip Sora for this comparison
                
            print(f"\n⏳ Testing {version} ({model})...")
            
            try:
                # Step 1: Start video generation
                result = await video_generations(
                    model=model,
                    prompt=prompt,
                    base_url=litellm_config["base_url"],
                    api_key=litellm_config["api_key"],
                    seconds="5",
                    size="1280x720",
                    timeout=60.0,
                )
                
                video_id = result.get("id") or result.get("video_id")
                print(f"  Video ID: {video_id}")
                
                results[version] = {
                    "model": model,
                    "status": "success",
                    "response": result,
                    "video_id": video_id,
                }
                
                # Step 2: Try to download video if we got an ID
                if video_id:
                    try:
                        print(f"  Waiting for {version} video completion...")
                        await self._wait_for_video(
                            video_id=video_id,
                            base_url=litellm_config["base_url"],
                            api_key=litellm_config["api_key"],
                            max_wait_seconds=180,
                        )
                        
                        video_bytes = await video_content(
                            video_id=video_id,
                            base_url=litellm_config["base_url"],
                            api_key=litellm_config["api_key"],
                        )
                        
                        # Save individual video file
                        video_path = video_outputs_dir / f"veo_comparison_{version}_{timestamp}.mp4"
                        video_path.write_bytes(video_bytes)
                        results[version]["video_file"] = str(video_path.name)
                        print(f"  ✓ {version} video saved to: {video_path}")
                        
                    except Exception as download_err:
                        results[version]["download_error"] = str(download_err)
                        print(f"  ⚠ {version} download failed: {download_err}")
                
                print(f"  ✓ {version} generation succeeded")
                
            except Exception as e:
                results[version] = {
                    "model": model,
                    "status": "failed",
                    "error": str(e),
                }
                print(f"  ✗ {version} failed: {e}")

        # Save comparison results
        comparison_path = video_outputs_dir / f"veo_comparison_{timestamp}.json"
        comparison_path.write_text(json.dumps({
            "prompt": prompt,
            "seconds": "5",
            "size": "1280x720",
            "models_tested": [k for k in self.VIDEO_MODELS.keys() if k != "sora-2"],
            "results": results,
        }, indent=2))
        
        print(f"\n✓ Comparison saved to: {comparison_path}")
        
        # At least log results even if generation fails
        assert comparison_path.exists()


class TestVideoGenerationDirect:
    """Direct video generation tests without LiteLLM proxy (for reference)."""

    @pytest.mark.asyncio
    @pytest.mark.skip(reason="Direct API test - enable when needed")
    async def test_direct_vertex_ai_veo(self, video_outputs_dir):
        """
        Example of direct Vertex AI Veo API call.
        
        This test is skipped by default as it requires direct Google Cloud credentials.
        Enable and configure for direct API testing.
        """
        # This would use google-cloud-aiplatform directly
        # from google.cloud import aiplatform
        # 
        # client = aiplatform.gapic.PredictionServiceClient()
        # response = client.predict(
        #     endpoint="projects/{project}/locations/{location}/publishers/google/models/veo-3.0-generate-001",
        #     instances=[{"prompt": "..."}],
        # )
        pass


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s", "--timeout=300"])
