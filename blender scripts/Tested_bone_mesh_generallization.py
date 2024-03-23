import bpy 
import numpy as np
import mathutils
from mathutils import Matrix 

# looks the mesh to find the bones which are related with the mesh
def takeBones(armature,mesh_obj):
    bones = []
    vertex_group_names = [group.name for group in mesh_obj.vertex_groups]
    for bone in armature.data.bones:
        if bone.name not in vertex_group_names:
            continue
        bones.append(bone)
    return bones

def findLocalMatrices():
    local_matrices=[]
    for bone in bones:
        bone_matrix_local = bone.matrix_local
        local_matrices.append(bone_matrix_local)
    return local_matrices

def matrix_world(armature_ob, bone_name):
    local = armature_ob.data.bones[bone_name].matrix_local
    basis = armature_ob.pose.bones[bone_name].matrix_basis

    parent = armature_ob.pose.bones[bone_name].parent
    #print(" this is the parent -> ",parent)
    if parent == None:
        return  local @ basis
    else:
        parent_local = armature_ob.data.bones[parent.name].matrix_local
        return matrix_world(armature_ob, parent.name) @ (parent_local.inverted() @ local) @ basis

def calculate_bones_matrices(bones,armature):
    final_matrices = []
    armature_matrix_world = armature.matrix_world
    for bone in bones:
        bone_matrix_world = armature_matrix_world @ matrix_world(armature,bone.name)
        final_matrices.append(bone_matrix_world)

    return final_matrices

def crop_matrix(matrix_list):
    cropped = []
    for matrix in matrix_list:
        matrix_np = np.array(matrix)
        cropped_matrix = matrix_np[:3, :4]
        cropped.append(cropped_matrix)
    return cropped

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

def find_matrices_Frames(armature,bones):
    start_frame = 0 
    end_frame = 43
    final_matrices_full = []
    cropped_matrices_full = [] 

    for frame in range(start_frame, end_frame +1):
        bpy.context.scene.frame_set(frame)
        final_matrices_frame = calculate_bones_matrices(bones,armature)
        final_matrices_full.append(final_matrices_frame)

        final_matrices_frame_cropped = crop_matrix(final_matrices_frame)
        cropped_matrices_full.append(final_matrices_frame_cropped)
        
    # return the full matrices and the cropped matrices 
    return final_matrices_full,cropped_matrices_full

def find_mesh_vertices(final_matrices,visualize):
    
    local_matrices = findLocalMatrices()
    
    mesh_obj = bpy.data.objects["Alpha_Surface"]
    mesh = mesh_obj.data
    vertices_co = []

    for vertex in mesh.vertices:
        coordinates = np.array(vertex.co)
        transformed_coordinates = np.zeros(3)

        for group in vertex.groups: 
            bone_index = group.group
            weight = group.weight
            #print(weight)

            bone_matrix = final_matrices[bone_index]
            local_bone_matrix=local_matrices[bone_index]
            inv_local_bone_matrix =  np.linalg.inv(local_bone_matrix)

            vertex_coords_homogeneous  = np.append(coordinates, 1.0)
            #print(vertex_coords_homogeneous.shape)

            #borei na xreiastei prin to pollaplasiasmo na to kanw np.array giati einai matrix logika
            v_relative_homogeneous  =  inv_local_bone_matrix @ vertex_coords_homogeneous 

            # transformation ta vertices me bash ta bones 
            skinned_vertex_homogeneous = np.array(bone_matrix) @ v_relative_homogeneous #np.append(v_relative, 1.0)
            #print("skinned_vertex_homogeneous:" , skinned_vertex_homogeneous.shape)
            transformed_coordinates += skinned_vertex_homogeneous[:3] * weight 

        # Append transformed coordinates
        vertices_co.append(tuple(transformed_coordinates[:3]))
        
    vertices_array = np.array(vertices_co)
    if visualize == True:
        visualizeTheMeshVerices(vertices_array)
    return vertices_array

def fastWayCaclulateMesh(final_matrices,visualize):
    
    local_matrices= findLocalMatrices()
    mesh_obj = bpy.data.objects["Alpha_Surface"]
    mesh = mesh_obj.data
    vertices_co = []
    final_matrices = np.array(final_matrices)
    vertices_co = np.hstack((np.array([v.co for v in mesh.vertices]), np.ones((len(mesh.vertices), 1))))
    transformed_coordinates = np.zeros((len(mesh.vertices), 3))

    inv_local_matrices = np.linalg.inv(local_matrices)

    for vertex in mesh.vertices:
        weights = np.array([group.weight for group in vertex.groups])
        bone_indices = [group.group for group in vertex.groups]

        v_relative_homogeneous_all = vertices_co[vertex.index] @ inv_local_matrices[bone_indices].transpose((0, 2, 1))

        bone_matrices = final_matrices[bone_indices]

        transformed_vertices = (bone_matrices @ v_relative_homogeneous_all[..., None]).squeeze()

        transformed_vertex = (transformed_vertices * weights[..., None]).sum(axis=0)[:3]

        transformed_coordinates[vertex.index] = transformed_vertex


    vertices_array = np.array(transformed_coordinates)
    if visualize == True:
        visualizeTheMeshVerices(vertices_array)
    return vertices_array

def TestMeshVerices(visualize , final_full):
    if(visualize == True ):
        find_mesh_vertices(final_full[40],visualize)  #slow
        fastWayCaclulateMesh(final_full[0],visualize) #fast

def makeVectorsInsteadOFMatrices(cropped_final_full):
    flattened_frames = []
    for frame in cropped_final_full:
        flattened_bone_matrices = []
        for bone_matrix in frame:
            flattened_matrix = np.array(bone_matrix).flatten()
            flattened_bone_matrices.append(flattened_matrix)
        flattened_frames.append(flattened_bone_matrices)
    return flattened_frames

def make_frames_file(cropped_final_full,bones):
    
    vectors = makeVectorsInsteadOFMatrices(cropped_final_full) # gives all the frame vectors 
    with open("BonesForFrames.txt","w") as file:

        file.write("Number of bones: "+str(len(bones))+ "\n")
        file.write("frame: 0"+"\n")
        i = 1 
        for frame in vectors:
            if( i == len(vectors)): # borw na to bgalw 
                break 
            for bone_matrix in frame:
                bone_matrix_str = " ".join([str(num) for num in bone_matrix])
                file.write(bone_matrix_str +"\n")

            
            file.write("frame:" + str(i) + "\n")
            i +=1   

def mesh_for_all_frames(final_full):
    print("calculate mesh vertices")
    mesh_for_all_frames = [] 
    for frame in range(0,43):
        mesh_for_all_frames.append(fastWayCaclulateMesh(final_full[frame],False))
    return mesh_for_all_frames


def makeMeshVertices(mesh_for_all_frames):
    print("Making txt files for mesh vertices of hole animation")
    all_mesh = np.array(mesh_for_all_frames)
    np.save("all_mesh_vertices.npy", all_mesh)


#def loadMeshVertices(file_path):
#    all_mesh_frames = np.load(file_path)
#    return all_mesh_frames
    
armature = bpy.data.objects['Armature']
mesh_obj = bpy.data.objects["Alpha_Surface"]
# make it true if you want to visualize the mesh vertices in the function find_mesh_vertices, fastWayCalculateMesh 
visualize = True 
num_frames = 82 

# prepei na skiparw ta final_matrices pou den exoun sxesh me to mesh 
bones = takeBones(armature,mesh_obj)

#txt for bones
final_full, cropped_final_full = find_matrices_Frames(armature,bones)
make_frames_file(cropped_final_full,bones)

# txt for  mesh gia olo to animation
mesh_full= mesh_for_all_frames(final_full) 
makeMeshVertices(mesh_full)

#loaded_full_mesh  = loadMeshVertices(num_frames)
#print(loaded_full_mesh)