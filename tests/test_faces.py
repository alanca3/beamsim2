"""Headless unit tests for geometry.faces — face-local driver placement math.

All tests are CI-safe: no VTK, no Qt, no gmsh, no NumCalc required.

Build-order item 10 follow-up (interactive driver placement).
"""

from __future__ import annotations

import numpy as np
import pytest

from beamsim2.geometry.faces import (
    FacePlacement,
    all_face_bases,
    clamp_uv_to_face,
    classify_face,
    face_basis,
    face_local_to_center,
    face_local_to_spec,
    fits_on_face,
    validate_spec_on_box,
    world_to_face_uv,
)

# ---------------------------------------------------------------------------
# Box used across most tests
# ---------------------------------------------------------------------------
W, H, D = 0.12, 0.10, 0.08  # default box in metres


# ---------------------------------------------------------------------------
# face_basis: exact table values
# ---------------------------------------------------------------------------


class TestFaceBasis:
    """face_basis returns the exact values from the module coordinate table."""

    def test_face0_plus_z(self):
        b = face_basis(0, W, H, D)
        assert b.face_id == 0
        assert b.normal == (0.0, 0.0, 1.0)
        assert b.centroid == pytest.approx((W / 2, H / 2, D))
        assert b.u_hat == (1.0, 0.0, 0.0)
        assert b.v_hat == (0.0, 1.0, 0.0)
        assert b.half_u == pytest.approx(W / 2)
        assert b.half_v == pytest.approx(H / 2)

    def test_face1_minus_z(self):
        b = face_basis(1, W, H, D)
        assert b.normal == (0.0, 0.0, -1.0)
        assert b.centroid == pytest.approx((W / 2, H / 2, 0.0))
        assert b.u_hat == (1.0, 0.0, 0.0)
        assert b.v_hat == (0.0, 1.0, 0.0)
        assert b.half_u == pytest.approx(W / 2)
        assert b.half_v == pytest.approx(H / 2)

    def test_face2_plus_x(self):
        b = face_basis(2, W, H, D)
        assert b.normal == (1.0, 0.0, 0.0)
        assert b.centroid == pytest.approx((W, H / 2, D / 2))
        assert b.u_hat == (0.0, 1.0, 0.0)
        assert b.v_hat == (0.0, 0.0, 1.0)
        assert b.half_u == pytest.approx(H / 2)
        assert b.half_v == pytest.approx(D / 2)

    def test_face3_minus_x(self):
        b = face_basis(3, W, H, D)
        assert b.normal == (-1.0, 0.0, 0.0)
        assert b.centroid == pytest.approx((0.0, H / 2, D / 2))
        assert b.u_hat == (0.0, 1.0, 0.0)
        assert b.v_hat == (0.0, 0.0, 1.0)

    def test_face4_plus_y(self):
        b = face_basis(4, W, H, D)
        assert b.normal == (0.0, 1.0, 0.0)
        assert b.centroid == pytest.approx((W / 2, H, D / 2))
        assert b.u_hat == (1.0, 0.0, 0.0)
        assert b.v_hat == (0.0, 0.0, 1.0)
        assert b.half_u == pytest.approx(W / 2)
        assert b.half_v == pytest.approx(D / 2)

    def test_face5_minus_y(self):
        b = face_basis(5, W, H, D)
        assert b.normal == (0.0, -1.0, 0.0)
        assert b.centroid == pytest.approx((W / 2, 0.0, D / 2))

    def test_invalid_face_id(self):
        with pytest.raises(ValueError, match="face_id must be 0..5"):
            face_basis(6, W, H, D)

    def test_all_face_bases_length(self):
        bases = all_face_bases(W, H, D)
        assert len(bases) == 6
        for i, b in enumerate(bases):
            assert b.face_id == i

    def test_u_hat_v_hat_orthonormal(self):
        """u_hat and v_hat are unit vectors orthogonal to normal and to each other."""
        for i in range(6):
            b = face_basis(i, W, H, D)
            u = np.array(b.u_hat)
            v = np.array(b.v_hat)
            n = np.array(b.normal)
            assert np.linalg.norm(u) == pytest.approx(1.0)
            assert np.linalg.norm(v) == pytest.approx(1.0)
            assert np.dot(u, v) == pytest.approx(0.0, abs=1e-12)
            assert np.dot(u, n) == pytest.approx(0.0, abs=1e-12)
            assert np.dot(v, n) == pytest.approx(0.0, abs=1e-12)


# ---------------------------------------------------------------------------
# face_local_to_center: on-plane guarantee
# ---------------------------------------------------------------------------


class TestFaceLocalToCenter:
    def test_zero_offset_is_centroid(self):
        """u=v=0 must land exactly on the face centroid."""
        for fid in range(6):
            fp = FacePlacement(face_id=fid, u=0.0, v=0.0, radius=0.02)
            center = face_local_to_center(fp, W, H, D)
            b = face_basis(fid, W, H, D)
            assert center == pytest.approx(b.centroid)

    def test_center_lies_on_face_plane(self):
        """For any (u,v), dot(center - centroid, normal) == 0."""
        offsets = [(0.0, 0.0), (0.01, 0.0), (0.0, 0.01), (-0.02, 0.015)]
        for fid in range(6):
            b = face_basis(fid, W, H, D)
            centroid = np.array(b.centroid)
            normal = np.array(b.normal)
            for u, v in offsets:
                fp = FacePlacement(face_id=fid, u=u, v=v, radius=0.01)
                center = np.array(face_local_to_center(fp, W, H, D))
                dist = abs(float(np.dot(center - centroid, normal)))
                assert dist < 1e-14, f"face {fid} (u={u},v={v}): dist={dist}"

    def test_existing_default_spec(self):
        """The legacy default center (0.06, 0.05, 0.08) on a 0.12×0.10×0.08 box
        is face 0 at u=v=0."""
        fp = FacePlacement(face_id=0, u=0.0, v=0.0, radius=0.020)
        center = face_local_to_center(fp, 0.12, 0.10, 0.08)
        assert center == pytest.approx((0.06, 0.05, 0.08))

    def test_v5_driver_a(self):
        """V-5 driver A spec: center=(0.035, 0.05, 0.08) is face 0, u=-0.025, v=0."""
        fp = FacePlacement(face_id=0, u=-0.025, v=0.0, radius=0.020)
        center = face_local_to_center(fp, 0.12, 0.10, 0.08)
        assert center == pytest.approx((0.035, 0.05, 0.08), abs=1e-12)


# ---------------------------------------------------------------------------
# face_local_to_spec
# ---------------------------------------------------------------------------


class TestFaceLocalToSpec:
    def test_spec_center_is_derived_center(self):
        fp = FacePlacement(face_id=0, u=0.01, v=-0.01, radius=0.015)
        spec = face_local_to_spec(fp, W, H, D)
        expected = face_local_to_center(fp, W, H, D)
        assert spec.center == pytest.approx(expected)

    def test_spec_normal_matches_face(self):
        for fid in range(6):
            fp = FacePlacement(face_id=fid, u=0.0, v=0.0, radius=0.01)
            spec = face_local_to_spec(fp, W, H, D)
            b = face_basis(fid, W, H, D)
            assert spec.normal == b.normal

    def test_spec_radius(self):
        fp = FacePlacement(face_id=2, u=0.0, v=0.0, radius=0.030)
        spec = face_local_to_spec(fp, W, H, D)
        assert spec.radius == pytest.approx(0.030)


# ---------------------------------------------------------------------------
# world_to_face_uv round-trip
# ---------------------------------------------------------------------------


class TestWorldToFaceUVRoundTrip:
    def test_round_trip(self):
        """world_to_face_uv(face_local_to_center(fp)) == (fp.u, fp.v)."""
        test_cases = [
            (0, 0.0, 0.0),
            (0, 0.02, -0.01),
            (1, -0.03, 0.02),
            (2, 0.01, 0.02),
            (3, 0.0, -0.01),
            (4, 0.01, 0.0),
            (5, -0.02, 0.01),
        ]
        for fid, u, v in test_cases:
            fp = FacePlacement(face_id=fid, u=u, v=v, radius=0.01)
            center = face_local_to_center(fp, W, H, D)
            u_rt, v_rt = world_to_face_uv(fid, center, W, H, D)
            assert u_rt == pytest.approx(u, abs=1e-13), f"face {fid}: u round-trip"
            assert v_rt == pytest.approx(v, abs=1e-13), f"face {fid}: v round-trip"


# ---------------------------------------------------------------------------
# fits_on_face
# ---------------------------------------------------------------------------


class TestFitsOnFace:
    def test_centred_small_fits(self):
        fp = FacePlacement(face_id=0, u=0.0, v=0.0, radius=0.01)
        assert fits_on_face(fp, W, H, D)

    def test_centred_just_fits_both(self):
        # radius == min(half_u, half_v) → |0|+r ≤ half_u and ≤ half_v → fits
        b = face_basis(0, W, H, D)
        max_r = min(b.half_u, b.half_v)  # 0.05 for face 0 (H/2 < W/2)
        fp = FacePlacement(face_id=0, u=0.0, v=0.0, radius=max_r)
        assert fits_on_face(fp, W, H, D)

    def test_barely_overflows_u(self):
        b = face_basis(0, W, H, D)
        # Overflow in v direction (half_v=0.05 < half_u=0.06)
        fp = FacePlacement(face_id=0, u=0.0, v=0.0, radius=b.half_v + 0.0001)
        assert not fits_on_face(fp, W, H, D)

    def test_offset_too_large(self):
        b = face_basis(0, W, H, D)
        fp = FacePlacement(face_id=0, u=b.half_u, v=0.0, radius=0.001)
        # |u| + r = half_u + 0.001 > half_u
        assert not fits_on_face(fp, W, H, D)

    def test_fits_on_all_faces_at_centroid(self):
        for fid in range(6):
            fp = FacePlacement(face_id=fid, u=0.0, v=0.0, radius=0.005)
            assert fits_on_face(fp, W, H, D), f"face {fid} centroid should fit"


# ---------------------------------------------------------------------------
# clamp_uv_to_face
# ---------------------------------------------------------------------------


class TestClampUVToFace:
    def test_already_inside_unchanged(self):
        u, v = clamp_uv_to_face(0, 0.01, 0.01, 0.01, W, H, D)
        assert u == pytest.approx(0.01)
        assert v == pytest.approx(0.01)

    def test_clamps_u_overflow(self):
        b = face_basis(0, W, H, D)
        u, v = clamp_uv_to_face(0, b.half_u + 0.05, 0.0, 0.01, W, H, D)
        assert abs(u) + 0.01 <= b.half_u + 1e-12

    def test_clamps_negative(self):
        b = face_basis(0, W, H, D)
        u, v = clamp_uv_to_face(0, -(b.half_u + 0.05), 0.0, 0.01, W, H, D)
        assert u < 0
        assert abs(u) + 0.01 <= b.half_u + 1e-12

    def test_disk_too_large_clamps_to_centroid(self):
        # radius > half_u: lim_u = max(0, half_u - radius) = 0 → always u=0
        b = face_basis(0, W, H, D)
        u, v = clamp_uv_to_face(0, 0.05, 0.0, b.half_u + 0.01, W, H, D)
        assert u == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# classify_face
# ---------------------------------------------------------------------------


class TestClassifyFace:
    def test_classify_by_normal_hint(self):
        """With a clean axis-aligned normal hint, classification is exact."""
        normals_expected = [
            ((0, 0, 1), 0),
            ((0, 0, -1), 1),
            ((1, 0, 0), 2),
            ((-1, 0, 0), 3),
            ((0, 1, 0), 4),
            ((0, -1, 0), 5),
        ]
        for n, expected_fid in normals_expected:
            # Point on the corresponding face centroid
            b = face_basis(expected_fid, W, H, D)
            fid = classify_face(b.centroid, n, W, H, D)
            assert fid == expected_fid, f"normal {n}: expected {expected_fid}, got {fid}"

    def test_classify_by_point_no_hint(self):
        """Without a normal hint, classification selects the nearest face."""
        # A point on the +z face plane
        fid = classify_face((W / 2, H / 2, D), None, W, H, D)
        assert fid == 0  # +z face

        fid = classify_face((W / 2, H / 2, 0.0), None, W, H, D)
        assert fid == 1  # -z face

    def test_classify_near_face_2(self):
        fid = classify_face((W, H / 2, D / 2), None, W, H, D)
        assert fid == 2  # +x face


# ---------------------------------------------------------------------------
# validate_spec_on_box
# ---------------------------------------------------------------------------


class TestValidateSpecOnBox:
    def test_valid_returns_none(self):
        """All existing test specs must pass validation."""
        # V-5 driver A on 0.12×0.10×0.08 box, +z face
        msg = validate_spec_on_box((0.035, 0.05, 0.08), (0, 0, 1), 0.020, 0.12, 0.10, 0.08)
        assert msg is None

        # V-5 driver B
        msg = validate_spec_on_box((0.085, 0.05, 0.08), (0, 0, 1), 0.020, 0.12, 0.10, 0.08)
        assert msg is None

        # geometry_health test: -z face of 0.2×0.15×0.10 box
        msg = validate_spec_on_box((0.1, 0.075, 0.0), (0, 0, -1), 0.030, 0.20, 0.15, 0.10)
        assert msg is None

        # stage1 test: -z face of 0.2×0.3×0.2 box
        msg = validate_spec_on_box((0.10, 0.20, 0.0), (0, 0, -1), 0.075, 0.20, 0.30, 0.20)
        assert msg is None

    def test_off_plane_returns_message(self):
        # center z=0.079 is 1mm off the z=0.08 face
        msg = validate_spec_on_box((0.06, 0.05, 0.079), (0, 0, 1), 0.020, 0.12, 0.10, 0.08)
        assert msg is not None
        assert "off" in msg.lower() or "mm" in msg.lower()

    def test_oversized_disk_returns_message(self):
        # r=0.07 on the +z face of 0.12×0.10 box: half_u=0.06, so |u|+r=0.07>0.06
        msg = validate_spec_on_box((0.06, 0.05, 0.08), (0, 0, 1), 0.070, 0.12, 0.10, 0.08)
        assert msg is not None
        assert "overflow" in msg.lower() or "exceed" in msg.lower() or ">" in msg

    def test_non_axis_normal_returns_message(self):
        msg = validate_spec_on_box(
            (0.06, 0.05, 0.08), (0.0, 0.7071, 0.7071), 0.020, 0.12, 0.10, 0.08
        )
        assert msg is not None
        assert "axis-aligned" in msg.lower() or "does not match" in msg.lower()

    def test_zero_normal_returns_message(self):
        msg = validate_spec_on_box((0.06, 0.05, 0.08), (0, 0, 0), 0.020, 0.12, 0.10, 0.08)
        assert msg is not None
        assert "zero" in msg.lower()

    def test_just_touching_edge_is_valid(self):
        # r = min(half_u, half_v) → disk exactly inscribed in face, should pass
        b = face_basis(0, W, H, D)
        max_r = min(b.half_u, b.half_v)  # 0.05
        fp = FacePlacement(face_id=0, u=0.0, v=0.0, radius=max_r)
        center = face_local_to_center(fp, W, H, D)
        msg = validate_spec_on_box(center, (0, 0, 1), max_r, W, H, D)
        assert msg is None
