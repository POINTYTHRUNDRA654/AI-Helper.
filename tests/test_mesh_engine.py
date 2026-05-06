"""Tests for ai_helper.mesh_engine."""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from ai_helper.mesh_engine import (
    Fallout4MeshKnowledge,
    FreeImage3DClient,
    FreeImage3DResult,
    FREE_IMAGE_TO_3D_BACKENDS,
    HuggingFaceDepthEstimator,
    MeshEngine,
    MeshValidator,
    ValidationResult,
    MeshyClient,
    MeshyTaskResult,
)


# ---------------------------------------------------------------------------
# Fallout4MeshKnowledge
# ---------------------------------------------------------------------------


class TestFallout4MeshKnowledge(unittest.TestCase):
    def setUp(self):
        self.kb = Fallout4MeshKnowledge()

    def test_answer_polygon_question(self):
        answer = self.kb.answer("How many polygons can a weapon mesh use?")
        self.assertIn("weapon", answer.lower())
        self.assertIn("5000", answer)

    def test_answer_nif_version(self):
        answer = self.kb.answer("What NIF version does Fallout 4 use?")
        self.assertIn("20.2.0.7", answer)

    def test_answer_texture_channels(self):
        answer = self.kb.answer("What texture channels does Fallout 4 use?")
        self.assertIn("_d.dds", answer)
        self.assertIn("_n.dds", answer)

    def test_answer_lod(self):
        answer = self.kb.answer("Tell me about LOD tiers")
        self.assertIn("LOD", answer)

    def test_answer_collision(self):
        answer = self.kb.answer("What collision shapes should I use?")
        self.assertIn("bhk", answer.lower())

    def test_answer_workflow(self):
        answer = self.kb.answer("What is the workflow to convert images to meshes?")
        # Should mention free tools and TripoSG
        self.assertIn("TripoSG", answer)

    def test_answer_free_tools(self):
        answer = self.kb.answer("What free tools can I use?")
        self.assertIn("github", answer.lower())

    def test_answer_generic_overview(self):
        answer = self.kb.answer("xyzzy totally unrelated query")
        self.assertIn("Fallout 4", answer)

    def test_validate_mesh_metadata_within_budget(self):
        issues = self.kb.validate_mesh_metadata(3000, "weapon")
        self.assertEqual(issues, [])

    def test_validate_mesh_metadata_over_budget(self):
        issues = self.kb.validate_mesh_metadata(99999, "weapon")
        self.assertTrue(len(issues) > 0)
        self.assertIn("5,000", issues[0])

    def test_validate_mesh_metadata_unknown_type(self):
        issues = self.kb.validate_mesh_metadata(100, "unknown_asset_xyz")
        self.assertTrue(len(issues) > 0)

    def test_get_workflow_returns_list(self):
        steps = self.kb.get_workflow()
        self.assertIsInstance(steps, list)
        self.assertGreater(len(steps), 5)

    def test_get_nif_export_settings(self):
        settings = self.kb.get_nif_export_settings()
        self.assertEqual(settings["version"], "20.2.0.7")
        self.assertEqual(settings["user_version"], 12)

    def test_get_texture_requirements(self):
        req = self.kb.get_texture_requirements()
        self.assertIn("diffuse", req)
        self.assertIn("normal", req)

    def test_get_polygon_budget_known(self):
        self.assertEqual(self.kb.get_polygon_budget("weapon"), 5000)

    def test_get_polygon_budget_unknown(self):
        self.assertIsNone(self.kb.get_polygon_budget("does_not_exist"))


# ---------------------------------------------------------------------------
# FreeImage3DClient — backend registry and install instructions
# ---------------------------------------------------------------------------


class TestFreeImage3DClientInfo(unittest.TestCase):
    def setUp(self):
        self.client = FreeImage3DClient()

    def test_list_backends_contains_triposg(self):
        text = FreeImage3DClient.list_backends()
        self.assertIn("triposg", text)
        self.assertIn("TripoSG", text)
        self.assertIn("MIT", text)

    def test_list_backends_contains_all_keys(self):
        text = FreeImage3DClient.list_backends()
        for key in FREE_IMAGE_TO_3D_BACKENDS:
            self.assertIn(key, text)

    def test_list_backends_notes_meshy_is_commercial(self):
        text = FreeImage3DClient.list_backends()
        self.assertIn("commercial", text.lower())

    def test_install_instructions_triposg(self):
        text = FreeImage3DClient.install_instructions("triposg")
        self.assertIn("github.com/VAST-AI-Research/TripoSG", text)
        self.assertIn("MIT", text)
        self.assertIn("pip install", text)

    def test_install_instructions_trellis(self):
        text = FreeImage3DClient.install_instructions("trellis")
        self.assertIn("microsoft/TRELLIS", text)

    def test_install_instructions_unknown_backend(self):
        text = FreeImage3DClient.install_instructions("nonexistent_backend")
        self.assertIn("Unknown backend", text)

    def test_backends_registry_has_required_fields(self):
        for key, info in FREE_IMAGE_TO_3D_BACKENDS.items():
            with self.subTest(backend=key):
                self.assertIn("github", info)
                self.assertIn("license", info)
                self.assertIn("vram_gb", info)
                self.assertIn("install", info)
                # All should be free (MIT or Apache)
                self.assertIn(info["license"], ("MIT", "Apache 2.0"))

    def test_triposg_is_recommended_default(self):
        """TripoSG should be first and marked as recommended."""
        keys = list(FREE_IMAGE_TO_3D_BACKENDS.keys())
        self.assertEqual(keys[0], "triposg")

    def test_run_triposg_missing_repo_returns_error(self):
        client = FreeImage3DClient(repo_root="/nonexistent/path/xyz")
        result = client.run_triposg("/tmp/fake.jpg", "/tmp/output")
        self.assertFalse(result.success)
        self.assertIn("TripoSG", result.error)
        self.assertIn("git clone", result.error)

    def test_run_trellis_missing_repo_returns_error(self):
        client = FreeImage3DClient(repo_root="/nonexistent/path/xyz")
        result = client.run_trellis("/tmp/fake.jpg", "/tmp/output")
        self.assertFalse(result.success)
        self.assertIn("TRELLIS", result.error)

    def test_run_triposr_missing_repo_returns_error(self):
        client = FreeImage3DClient(repo_root="/nonexistent/path/xyz")
        result = client.run_triposr("/tmp/fake.jpg", "/tmp/output")
        self.assertFalse(result.success)
        self.assertIn("TripoSR", result.error)

    def test_run_shap_e_import_error_returns_error(self):
        client = FreeImage3DClient(repo_root="/nonexistent/path/xyz")
        result = client.run_shap_e("/tmp/fake.jpg", "/tmp/output")
        self.assertFalse(result.success)
        self.assertIn("Shap-E", result.error)


# ---------------------------------------------------------------------------
# FreeImage3DResult
# ---------------------------------------------------------------------------


class TestFreeImage3DResult(unittest.TestCase):
    def test_summary_success(self):
        r = FreeImage3DResult(
            backend="triposg",
            input_image="/tmp/photo.jpg",
            output_dir="/tmp/out",
            mesh_path="/tmp/out/mesh.glb",
            exported_files=["/tmp/out/mesh.glb"],
            elapsed_s=12.3,
            success=True,
        )
        summary = r.summary
        self.assertIn("✓", summary)
        self.assertIn("mesh.glb", summary)
        self.assertIn("12.3", summary)

    def test_summary_failure(self):
        r = FreeImage3DResult(
            backend="triposg",
            input_image="/tmp/photo.jpg",
            output_dir="/tmp/out",
            success=False,
            error="Something went wrong",
        )
        self.assertIn("✗", r.summary)
        self.assertIn("Something went wrong", r.summary)

    def test_summary_uses_backend_name(self):
        r = FreeImage3DResult(
            backend="triposg",
            input_image="/tmp/photo.jpg",
            output_dir="/tmp/out",
            success=True,
        )
        self.assertIn("TripoSG", r.summary)


# ---------------------------------------------------------------------------
# MeshyClient
# ---------------------------------------------------------------------------


class TestMeshyClient(unittest.TestCase):
    def test_no_api_key_raises(self):
        client = MeshyClient(api_key="")
        # Remove env var if present
        with patch.dict(os.environ, {"MESHY_API_KEY": ""}, clear=False):
            with self.assertRaises(ValueError):
                client.image_to_3d("/tmp/fake.jpg")

    def test_parse_task_succeeded(self):
        data = {
            "id": "abc123",
            "status": "SUCCEEDED",
            "model_urls": {"obj": "https://example.com/mesh.obj"},
            "progress": 100,
        }
        task = MeshyClient._parse_task(data)
        self.assertTrue(task.succeeded)
        self.assertEqual(task.task_id, "abc123")
        self.assertEqual(task.model_urls["obj"], "https://example.com/mesh.obj")

    def test_parse_task_failed(self):
        data = {"id": "xyz", "status": "FAILED", "message": "out of credits", "progress": 0}
        task = MeshyClient._parse_task(data)
        self.assertFalse(task.succeeded)
        self.assertEqual(task.error, "out of credits")

    def test_meshy_task_result_succeeded_property(self):
        t = MeshyTaskResult(task_id="1", status="SUCCEEDED")
        self.assertTrue(t.succeeded)
        t2 = MeshyTaskResult(task_id="2", status="FAILED")
        self.assertFalse(t2.succeeded)


# ---------------------------------------------------------------------------
# HuggingFaceDepthEstimator
# ---------------------------------------------------------------------------


class TestHuggingFaceDepthEstimator(unittest.TestCase):
    def test_list_free_models(self):
        text = HuggingFaceDepthEstimator.list_free_models()
        self.assertIn("Depth-Anything", text)
        self.assertIn("HuggingFace", text)
        self.assertIn("free", text.lower())

    def test_default_model_is_depth_anything(self):
        est = HuggingFaceDepthEstimator()
        self.assertIn("depth-anything", est.model_id.lower())

    def test_custom_model_id(self):
        est = HuggingFaceDepthEstimator(model_id="Intel/dpt-large")
        self.assertEqual(est.model_id, "Intel/dpt-large")

    def test_auto_device_returns_string(self):
        device = HuggingFaceDepthEstimator._auto_device()
        self.assertIn(device, ("cuda", "cpu"))

    def test_estimate_raises_import_error_without_transformers(self):
        est = HuggingFaceDepthEstimator()
        with patch.dict("sys.modules", {"transformers": None}):
            with self.assertRaises(ImportError):
                est._pipeline = None  # reset cached pipeline
                est.estimate("/tmp/fake.jpg")

    def test_save_depth_map(self):
        try:
            import numpy as np  # noqa: PLC0415
            from PIL import Image  # noqa: PLC0415
        except ImportError:
            self.skipTest("numpy/Pillow not installed")

        import numpy as np  # noqa: PLC0415, F811
        est = HuggingFaceDepthEstimator()
        depth = np.zeros((64, 64), dtype=np.float32)
        with tempfile.TemporaryDirectory() as tmp:
            path = est.save_depth_map(depth, str(Path(tmp) / "depth.png"))
            self.assertTrue(Path(path).exists())


# ---------------------------------------------------------------------------
# MeshValidator
# ---------------------------------------------------------------------------


class TestMeshValidator(unittest.TestCase):
    def test_nonexistent_file(self):
        validator = MeshValidator()
        result = validator.validate("/nonexistent/file.obj")
        self.assertFalse(result.passed)
        self.assertTrue(any("not found" in i.lower() for i in result.issues))

    def test_obj_stats_parser(self):
        with tempfile.NamedTemporaryFile(suffix=".obj", mode="w",
                                        delete=False, encoding="utf-8") as fh:
            fh.write("v 0 0 0\nv 1 0 0\nv 0 1 0\nvn 0 0 1\nvt 0 0\nf 1 2 3\n")
            obj_path = fh.name
        try:
            f, v, has_normals, has_uvs = MeshValidator._parse_obj_stats(Path(obj_path))
            self.assertEqual(v, 3)
            self.assertTrue(has_normals)
            self.assertTrue(has_uvs)
        finally:
            os.unlink(obj_path)

    def test_validation_result_str(self):
        r = ValidationResult(
            path="/tmp/mesh.obj",
            poly_count=500,
            vertex_count=300,
            has_normals=True,
            has_uvs=True,
            is_watertight=False,
        )
        text = str(r)
        self.assertIn("PASS", text)
        self.assertIn("500", text)

    def test_validation_result_fail(self):
        r = ValidationResult(
            path="/tmp/mesh.obj",
            poly_count=500,
            vertex_count=300,
            has_normals=True,
            has_uvs=False,
            is_watertight=False,
            issues=["No UV coordinates"],
        )
        self.assertFalse(r.passed)
        self.assertIn("FAIL", str(r))


# ---------------------------------------------------------------------------
# MeshEngine façade
# ---------------------------------------------------------------------------


class TestMeshEngine(unittest.TestCase):
    def setUp(self):
        self.engine = MeshEngine()

    def test_ask_knowledge_polygon(self):
        answer = self.engine.ask_knowledge("polygon budget for weapons")
        self.assertIn("5000", answer)

    def test_get_workflow(self):
        workflow = self.engine.get_workflow()
        self.assertIn("TripoSG", workflow)
        self.assertIn("Recommended", workflow)

    def test_list_free_tools(self):
        text = self.engine.list_free_tools()
        self.assertIn("triposg", text)
        self.assertIn("MIT", text)

    def test_install_instructions_triposg(self):
        text = self.engine.install_instructions("triposg")
        self.assertIn("TripoSG", text)
        self.assertIn("git clone", text)

    def test_install_instructions_unknown(self):
        text = self.engine.install_instructions("bad_backend")
        self.assertIn("Unknown", text)

    def test_free_image_to_3d_unknown_backend(self):
        result = self.engine.free_image_to_3d(
            "/tmp/fake.jpg", backend="nonexistent_backend"
        )
        self.assertFalse(result.success)
        self.assertIn("Unknown backend", result.error)

    def test_free_image_to_3d_triposg_no_repo(self):
        """Without TripoSG cloned, should return a clear error with install instructions."""
        engine = MeshEngine(free_models_root="/nonexistent/xyz")
        result = engine.free_image_to_3d("/tmp/fake.jpg", backend="triposg")
        self.assertFalse(result.success)
        self.assertIn("git clone", result.error)

    def test_validate_mesh_nonexistent(self):
        result = self.engine.validate_mesh("/nonexistent/mesh.obj")
        self.assertFalse(result.passed)

    def test_check_dependencies(self):
        text = self.engine.check_dependencies()
        self.assertIn("Mesh Engine", text)
        self.assertIn("opencv-python", text)
        self.assertIn("triposg", text)

    def test_meshy_no_key_errors_gracefully(self):
        engine = MeshEngine(meshy_api_key="")
        with patch.dict(os.environ, {"MESHY_API_KEY": ""}, clear=False):
            result = engine.meshy_image_to_3d("/tmp/fake.jpg")
        self.assertFalse(result.success)
        self.assertTrue(len(result.errors) > 0)

    def test_scan_images_no_images(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = self.engine.scan_images([], output_dir=tmp)
        self.assertFalse(result.success)
        self.assertTrue(any("No valid" in e for e in result.errors))

    def test_scan_images_missing_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = self.engine.scan_images(
                ["/nonexistent/a.jpg", "/nonexistent/b.jpg"],
                output_dir=tmp,
            )
        self.assertFalse(result.success)

    def test_scan_images_writes_metadata_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = self.engine.scan_images(
                ["/nonexistent/x.jpg"],
                output_dir=tmp,
            )
            meta_path = Path(tmp) / "scan_metadata.json"
            self.assertTrue(meta_path.exists())
            data = json.loads(meta_path.read_text())
            self.assertIn("input_images", data)
            self.assertIn("errors", data)


if __name__ == "__main__":
    unittest.main()
