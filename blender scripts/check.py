import bpy
import mathutils
import numpy as np

class Model(object):
	def __init__(self, name_of_obj_mesh, name_of_mesh, name_of_armature):
		self.name_of_obj_mesh = name_of_obj_mesh
		self.name_of_mesh = name_of_mesh
		self.name_of_armature = name_of_armature
		self.num_of_vertices = len(bpy.data.meshes[name_of_mesh].vertices)
		self.change_mesh_to_World_coordinates()

	def change_mesh_to_World_coordinates(self):
		obj = bpy.data.objects[self.name_of_obj_mesh]
		mat = obj.matrix_world
		mesh = bpy.data.meshes[self.name_of_mesh]
		mesh.transform(mat)
		obj.matrix_world = mathutils.Matrix()

	def get_mesh_v_in_all_frames(self):
		# get mesh vertices per frame
		scn = bpy.context.scene
		mesh_obj = bpy.data.objects[self.name_of_obj_mesh]
		start_frame, end_frame = self.__get_active_action_frame_end()
		print(start_frame,end_frame)
		vertices = np.array([v.co for v in bpy.data.meshes[self.name_of_mesh].vertices])
		print(vertices)
		for f in range(start_frame, end_frame + 1):
			bpy.context.scene.frame_set(f)
			temp_mesh = mesh_obj.to_mesh(preserve_all_data_layers=True)
			if f == 1:
				vertices = np.array([v.co for v in temp_mesh.vertices])
			else:
				vertices = np.vstack( (vertices, [v.co for v in temp_mesh.vertices]) )

			
		return vertices

	def get_mesh_v_of_whole_animation_3D(self):
		# get mesh vertices per frame
		scn = bpy.context.scene
		mesh_obj = bpy.data.objects[self.name_of_obj_mesh]
		start_frame, end_frame = self.__get_active_action_frame_end()

		vertices = []
		for v in range(0, self.num_of_vertices):
			vertices.append([])

		for f in range(start_frame, end_frame + 1):
			bpy.context.scene.frame_set(f)
			temp_mesh = mesh_obj.to_mesh()

			for v in range(0, self.num_of_vertices):
				vertices[v].append(np.array(temp_mesh.vertices[v].co))

			# bpy.data.meshes.remove(temp_mesh)
		return np.array(vertices)

	def __get_active_action_frame_end(self):
		action_name = self.__get_active_action_name()
		frame_end = bpy.data.actions[action_name].frame_range
		self.num_of_frames = int(frame_end[1])
		return int(frame_end[0]), int(frame_end[1])

	def __get_active_action_name(self):
		action_name = bpy.data.objects[self.name_of_armature].animation_data.action.name
		return action_name



model = Model("Alpha_Surface","Alpha_Surface","Armature")
vertices = model.get_mesh_v_in_all_frames()
print(vertices)