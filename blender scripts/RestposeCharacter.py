import bpy 
import numpy as np
import mathutils
from mathutils import Matrix 

def mesh_vertices_calc_rest_pose(mesh_obj):
    
    vertices = [vertex.co[:] for vertex in mesh_obj.data.vertices] 
    return np.array(vertices)

def visualizeTheMeshVerices(vertices_array):
    # Create a new mesh object to visualize the transformed vertices
    new_mesh = bpy.data.meshes.new("TransformedMesh")
    new_mesh.from_pydata(vertices_array, [], [])

    # Create a new object and link it to the scene
    new_obj = bpy.data.objects.new("TransformedObject", new_mesh)
    bpy.context.collection.objects.link(new_obj)

    # Set the new object as the active object and select it
    bpy.context.view_layer.objects.active = new_obj
    new_obj.select_set(True)

    # Switch to object mode and update the view
    bpy.ops.object.mode_set(mode='OBJECT')
    bpy.ops.object.select_all(action='DESELECT')
    new_obj.select_set(True)
    bpy.context.view_layer.objects.active = new_obj

def CreateRestPoseFile(rest_pose):
    print("RestPoseFile is created")
    np.save("RestPose.npy",rest_pose)

def main():
    mesh_obj = bpy.data.objects["Alpha_Surface"]    
    mesh_vertices = mesh_vertices_calc_rest_pose(mesh_obj)

    print(mesh_vertices)
    visualizeTheMeshVerices(mesh_vertices)
    CreateRestPoseFile(mesh_vertices)

if __name__ == "__main__":
    main()