#!/usr/bin/env python3
"""
Generate a realistic bottle mesh (STL file) for MuJoCo.

Usage:
    pip install numpy-stl
    python generate_bottle_mesh.py
"""

import numpy as np
from pathlib import Path

try:
    from stl import mesh
except ImportError:
    print("Installing numpy-stl...")
    import subprocess
    subprocess.check_call(["pip", "install", "numpy-stl"])
    from stl import mesh


def create_cylinder_vertices(radius, height, segments=32, z_offset=0):
    """Create vertices for a cylinder."""
    vertices = []
    
    # Create circles at top and bottom
    angles = np.linspace(0, 2*np.pi, segments, endpoint=False)
    
    bottom_center = [0, 0, z_offset]
    top_center = [0, 0, z_offset + height]
    
    bottom_ring = [[radius * np.cos(a), radius * np.sin(a), z_offset] for a in angles]
    top_ring = [[radius * np.cos(a), radius * np.sin(a), z_offset + height] for a in angles]
    
    return bottom_center, top_center, bottom_ring, top_ring, angles


def create_bottle_mesh():
    """Create a realistic bottle shape."""
    segments = 32
    faces = []
    
    # Bottle dimensions (in meters, scaled for MuJoCo)
    # Base
    base_radius = 0.038
    base_height = 0.015
    
    # Body
    body_radius = 0.035
    body_height = 0.12
    
    # Shoulder
    shoulder_radius = 0.025
    shoulder_height = 0.03
    
    # Neck
    neck_radius = 0.014
    neck_height = 0.05
    
    # Lip
    lip_radius = 0.016
    lip_height = 0.015
    
    # Generate profile points (radius, z)
    profile = [
        (0, 0),                                    # Bottom center
        (base_radius, 0),                          # Base outer bottom
        (base_radius, base_height),                # Base outer top
        (body_radius, base_height),                # Body start
        (body_radius, base_height + body_height),  # Body end
        (shoulder_radius, base_height + body_height + shoulder_height * 0.5),  # Shoulder
        (neck_radius, base_height + body_height + shoulder_height),  # Neck start
        (neck_radius, base_height + body_height + shoulder_height + neck_height),  # Neck end
        (lip_radius, base_height + body_height + shoulder_height + neck_height),  # Lip start
        (lip_radius, base_height + body_height + shoulder_height + neck_height + lip_height),  # Lip end
        (neck_radius - 0.002, base_height + body_height + shoulder_height + neck_height + lip_height),  # Inner lip
    ]
    
    # Create revolution surface
    angles = np.linspace(0, 2*np.pi, segments, endpoint=False)
    
    all_faces = []
    
    for i in range(len(profile) - 1):
        r1, z1 = profile[i]
        r2, z2 = profile[i + 1]
        
        for j in range(segments):
            j_next = (j + 1) % segments
            a1, a2 = angles[j], angles[j_next]
            
            # Four corners of quad
            p1 = [r1 * np.cos(a1), r1 * np.sin(a1), z1]
            p2 = [r1 * np.cos(a2), r1 * np.sin(a2), z1]
            p3 = [r2 * np.cos(a2), r2 * np.sin(a2), z2]
            p4 = [r2 * np.cos(a1), r2 * np.sin(a1), z2]
            
            # Two triangles per quad
            if r1 > 0.001 and r2 > 0.001:
                all_faces.append([p1, p2, p3])
                all_faces.append([p1, p3, p4])
            elif r1 > 0.001:  # Top cone
                all_faces.append([p1, p2, p3])
            elif r2 > 0.001:  # Bottom cone
                all_faces.append([p1, p3, p4])
    
    # Create bottom cap
    r_bottom = base_radius
    center = [0, 0, 0]
    for j in range(segments):
        j_next = (j + 1) % segments
        a1, a2 = angles[j], angles[j_next]
        p1 = [r_bottom * np.cos(a1), r_bottom * np.sin(a1), 0]
        p2 = [r_bottom * np.cos(a2), r_bottom * np.sin(a2), 0]
        all_faces.append([center, p2, p1])  # Reversed for correct normal
    
    # Convert to numpy mesh
    bottle = mesh.Mesh(np.zeros(len(all_faces), dtype=mesh.Mesh.dtype))
    for i, face in enumerate(all_faces):
        for j in range(3):
            bottle.vectors[i][j] = face[j]
    
    return bottle


def create_cap_mesh():
    """Create a bottle cap mesh."""
    segments = 32
    all_faces = []
    angles = np.linspace(0, 2*np.pi, segments, endpoint=False)
    
    # Cap dimensions
    outer_radius = 0.019
    inner_radius = 0.015
    height = 0.024
    top_thickness = 0.003
    
    # Outer wall
    for j in range(segments):
        j_next = (j + 1) % segments
        a1, a2 = angles[j], angles[j_next]
        
        p1 = [outer_radius * np.cos(a1), outer_radius * np.sin(a1), 0]
        p2 = [outer_radius * np.cos(a2), outer_radius * np.sin(a2), 0]
        p3 = [outer_radius * np.cos(a2), outer_radius * np.sin(a2), height]
        p4 = [outer_radius * np.cos(a1), outer_radius * np.sin(a1), height]
        
        all_faces.append([p1, p2, p3])
        all_faces.append([p1, p3, p4])
    
    # Top surface
    for j in range(segments):
        j_next = (j + 1) % segments
        a1, a2 = angles[j], angles[j_next]
        
        p1 = [outer_radius * np.cos(a1), outer_radius * np.sin(a1), height]
        p2 = [outer_radius * np.cos(a2), outer_radius * np.sin(a2), height]
        center = [0, 0, height]
        
        all_faces.append([p1, p2, center])
    
    # Inner wall (for grip inside)
    for j in range(segments):
        j_next = (j + 1) % segments
        a1, a2 = angles[j], angles[j_next]
        
        p1 = [inner_radius * np.cos(a1), inner_radius * np.sin(a1), 0]
        p2 = [inner_radius * np.cos(a2), inner_radius * np.sin(a2), 0]
        p3 = [inner_radius * np.cos(a2), inner_radius * np.sin(a2), height - top_thickness]
        p4 = [inner_radius * np.cos(a1), inner_radius * np.sin(a1), height - top_thickness]
        
        all_faces.append([p2, p1, p3])  # Reversed normal (inside)
        all_faces.append([p3, p1, p4])
    
    # Bottom ring (connects inner and outer)
    for j in range(segments):
        j_next = (j + 1) % segments
        a1, a2 = angles[j], angles[j_next]
        
        p1 = [outer_radius * np.cos(a1), outer_radius * np.sin(a1), 0]
        p2 = [outer_radius * np.cos(a2), outer_radius * np.sin(a2), 0]
        p3 = [inner_radius * np.cos(a2), inner_radius * np.sin(a2), 0]
        p4 = [inner_radius * np.cos(a1), inner_radius * np.sin(a1), 0]
        
        all_faces.append([p2, p1, p3])
        all_faces.append([p3, p1, p4])
    
    # Convert to numpy mesh
    cap = mesh.Mesh(np.zeros(len(all_faces), dtype=mesh.Mesh.dtype))
    for i, face in enumerate(all_faces):
        for j in range(3):
            cap.vectors[i][j] = face[j]
    
    return cap


def main():
    output_dir = Path(__file__).parent / "assets" / "meshes"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("Generating bottle mesh...")
    bottle = create_bottle_mesh()
    bottle_path = output_dir / "bottle.stl"
    bottle.save(str(bottle_path))
    print(f"  [OK] Saved: {bottle_path}")
    
    print("Generating cap mesh...")
    cap = create_cap_mesh()
    cap_path = output_dir / "cap.stl"
    cap.save(str(cap_path))
    print(f"  [OK] Saved: {cap_path}")
    
    print("\nMeshes generated! Now run:")
    print("   python view.py --task BottleCap-v0")


if __name__ == "__main__":
    main()
