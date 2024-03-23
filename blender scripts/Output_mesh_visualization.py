# this code works properly if the model script after training gives an npy file after training 
import bpy 
import numpy as np
import mathutils
from mathutils import Matrix 


def visualization_mesh(predicted_mesh,mesh_obj):
    mesh = mesh_obj.data
    #print(predicted_mesh)
    for i, vertex in enumerate(mesh.vertices):
        vertex.co = predicted_mesh[i]
    mesh.update()


#visualize the mesh which has the filepath, output visualization   
def visualize_mesh(filepath,mesh_obj_test,visualized_frame):

    all_vertices = np.load(filepath)

    vertices = len(all_vertices[1]) // 3

    reshaped_array_verices  = all_vertices.reshape(num_frames,vertices,3)

    visualization_mesh(reshaped_array_verices[visualized_frame],mesh_obj_test)


def calculateRestPose(mesh_target):

    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_all(action='SELECT')
    bpy.ops.mesh.quads_convert_to_tris()

    rest_pose = []

    for i in mesh_target.data.vertices:
        rest_pose.append(i.co)

    rest_pose = np.array(rest_pose)
    return rest_pose


def dot_visualizeTheMeshVerices(vertices_array):
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

mesh_obj = bpy.data.objects["Alpha_Surface"]
mesh_obj_test = bpy.data.objects["Surface"]
num_frames =  bpy.context.scene.frame_end +1 
visualized_frame = 0

#visualize_mesh("best_predicted_verices.npy",mesh_obj_test,visualized_frame)
rest_pose = calculateRestPose(mesh_obj)  # this is the rest pose of the mesh of my animation
print(rest_pose)
CreateRestPoseFile(rest_pose)
dot_visualizeTheMeshVerices(rest_pose)