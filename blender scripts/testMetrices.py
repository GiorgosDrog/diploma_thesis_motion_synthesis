import bpy 
import numpy as np
import mathutils
from mathutils import Matrix 


def takeBones(armature,mesh_obj):
    bones = []

    vertex_group_names = [group.name for group in mesh_obj.vertex_groups]
    for bone in armature.data.bones:
        if bone.name not in vertex_group_names:
            continue
        bones.append(bone)

    return bones

def findLocalMatrices(bones):
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

def find_matrices_Frames(armature,bones,last_frame):
    start_frame = 0 
    end_frame = last_frame
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


#This will make flatten frames to use it in the txt BoneForFrames.txt {it convert the final_full to np.array final full}
def makeVectorsInsteadOFMatrices(cropped_final_full):
    flattened_frames = []
    for frame in cropped_final_full:
        flattened_bone_matrices = []
        for bone_matrix in frame:
            flattened_matrix = np.array(bone_matrix).flatten()
            flattened_bone_matrices.append(flattened_matrix)
        flattened_frames.append(flattened_bone_matrices)
    return flattened_frames


def CreateBonesFile(cropped_final_full,filePath):
    FinalMatrices_nparray = makeVectorsInsteadOFMatrices(cropped_final_full)
    np.save(filePath,FinalMatrices_nparray)

#makes the meshVerticesFile
def makeMeshVertices(mesh_for_all_frames,filePath):
    print("Making np file for mesh vertices of hole animation (for all frames)")
    all_mesh = np.array(mesh_for_all_frames)
    np.save(filePath, all_mesh)

#mesh visualization
# ------------------------------------------------->
def find_mesh_vertices(final_matrices,mesh_obj,bones):
    local_matrices = findLocalMatrices(bones)
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
    return vertices_array



def fastWayCaclulateMesh(final_matrices,mesh_obj,bones):
    
    local_matrices= findLocalMatrices(bones)
    print(len(local_matrices))
    mesh = mesh_obj.data
    vertices_co = []
    final_matrices = np.array(final_matrices)
    print(len(final_matrices))
    vertices_co = np.hstack((np.array([v.co for v in mesh.vertices]), np.ones((len(mesh.vertices), 1))))
    transformed_coordinates = np.zeros((len(mesh.vertices), 3))

    inv_local_matrices = np.linalg.inv(local_matrices)

    for vertex in mesh.vertices:
        weights = np.array([group.weight for group in vertex.groups])
        # print(weights.shape)
        bone_indices = [group.group for group in vertex.groups]
        # print(len(bone_indices))
        v_relative_homogeneous_all = vertices_co[vertex.index] @ inv_local_matrices[bone_indices].transpose((0, 2, 1))

        bone_matrices = final_matrices[bone_indices]

        transformed_vertices = (bone_matrices @ v_relative_homogeneous_all[..., None]).squeeze()

        transformed_vertex = (transformed_vertices * weights[..., None]).sum(axis=0)[:3]

        transformed_coordinates[vertex.index] = transformed_vertex


    vertices_array = np.array(transformed_coordinates)
    return vertices_array


def mesh_for_all_frames(final_full,last_frame,mesh_obj,bones):
    print("calculate mesh vertices")
    mesh_for_all_frames = [] 
    for frame in range(0,last_frame+1):
        mesh_for_all_frames.append(find_mesh_vertices(final_full[frame],mesh_obj,bones))
    return mesh_for_all_frames

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

#filepath1 = bones filepath2 = mesh armature = armature_name mesh_obj = mesh_obj_name
def makeTheData(armature,mesh_obj,filepath1,filepath2):
    armature = bpy.data.objects[armature]
    mesh_obj = bpy.data.objects[mesh_obj]
    last_frame = bpy.context.scene.frame_end
    print(last_frame)
    bones_test = takeBones(armature,mesh_obj)
    final_full, cropped_final_full = find_matrices_Frames(armature,bones_test,last_frame)
    list = np.array(final_full)
    print(list[0])
    CreateBonesFile(cropped_final_full,filepath1)
    mesh_full= mesh_for_all_frames(final_full,last_frame,mesh_obj,bones_test) 
    makeMeshVertices(mesh_full,filepath2)
    print(np.array(mesh_full[0]))
    dot_visualizeTheMeshVerices(mesh_full[0])


# makeTheData("running","human","runningBones.npy","runningMesh.npy")
# makeTheData("jogging","human.001","joggingBones.npy","joggingMesh.npy")
# makeTheData("left_run","human.002","leftRunBones.npy","leftRunMesh.npy")
# makeTheData("low_run","human.003","lowRunBones.npy","lowRunMesh.npy")
# makeTheData("pistol_run","human.004","pistolRunBones.npy","pistolRunMesh.npy")
# makeTheData("dodge","human.005","dodge_bones.npy","dodge_mesh.npy")
# makeTheData("spring","human.006","spring_bones.npy","spring_mesh.npy")
# makeTheData("UpHands","human.007","Uphand_bones.npy","Uphand_mesh.npy")
# makeTheData("Hook","human.008","hook_bones.npy","hook_mesh.npy")
# makeTheData("climp","human.009","climp_bones.npy","climp_mesh.npy")
# makeTheData("Armature","human.010","Jump_bones.npy","Jump_mesh.npy")
# makeTheData("humanpistol","human.011","pistol_bones.npy","pistol_mesh.npy")

# makeTheData("lift_rinning","lift_running_mesh","lift_bones.npy","lift_mesh.npy")
# makeTheData("running1","running_mesh1","running1_bones.npy","running1_mesh.npy")
# makeTheData("running2","running_mesh2","running2_bones.npy","running2_mesh.npy")
# makeTheData("running3","running_mesh3","running3_bones.npy","running3_mesh.npy")
# makeTheData("running_Crou","running_Crou_mesh","running_Crou_bones.npy","running_Crou_mesh.npy")
    
# makeTheData("jogging_bones","jogging_mesh","jogging_bones.npy","jogging_mesh.npy")
# makeTheData("jump_forward","jump_forward_mesh","jump_forward_bones.npy","jump_forward_mesh.npy")
# makeTheData("jump_chill","jump_chillmesh","jump_chill_bones.npy","jump_chill_mesh.npy")
# makeTheData("jumping_down","jumping_down_mesh","jumping_down_bones.npy","jumping_down_mesh.npy")
    
makeTheData("Armature","human","jumpUp_bones.npy","jumpUp_mesh.npy")
makeTheData("Armature.001","human.001","jumpforward_bones.npy","jumpforward_mesh.npy")
makeTheData("Armature.002","human.002","jumpStatic_bones.npy","jumpStatic_mesh.npy")
makeTheData("Armature.003","human.003","jumpjoy_bones.npy","jumpjoy_mesh.npy")
makeTheData("Armature.004","human.004","jumpStatic2_bones.npy","jumpStatic2_mesh.npy")
makeTheData("Armature.005","human.005","jumpToBox_bones.npy","jumpToBox_mesh.npy")
makeTheData("Armature.006","human.006","jump2Run_bones.npy","jump2Run_mesh.npy")
makeTheData("Armature.007","human.007","jump3Run_bones.npy","jump3Run_mesh.npy")


