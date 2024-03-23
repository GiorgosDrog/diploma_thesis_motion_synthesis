import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torch.nn.utils import clip_grad_norm_

class Encoder_bones(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(Encoder_bones, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.layer_norm = nn.LayerNorm(hidden_size)
  
    def forward(self, input_data):
        self.lstm.flatten_parameters()
        output_encoder, (hidden_state, cell_state) = self.lstm(input_data)
        hidden_state = self.layer_norm(hidden_state)
        return output_encoder,(hidden_state, cell_state)
    
class Encoder_firstPose(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers,device):
        super(Encoder_firstPose, self).__init__()
        self.device= device
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.fc1 = nn.Linear(input_size ,hidden_size)
        self.fc2 = nn.Linear(hidden_size,hidden_size)

    def forward(self, input_data):
        x = F.tanh(self.fc1(input_data))
        x = F.tanh(self.fc2(x))
        return x 
    
class Decoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(Decoder, self).__init__()
        self.input_decoder = nn.Linear(hidden_size,hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True)
        self.output_lstm = nn.Linear(hidden_size,hidden_size)
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.fc = nn.Linear(hidden_size, output_size)
        

    def forward(self, decoder_input):
        self.lstm.flatten_parameters()
        output = self.input_decoder(decoder_input)
        output, _ = self.lstm(decoder_input)
        output = self.output_lstm(output)
        output = self.layer_norm(output)
        output = self.fc(output)
        return output

 
    
class Lstm_encoder_decoder(nn.Module):
  
    def __init__(self,input_size,hidden_size,num_layers,output_size,device):
        super(Lstm_encoder_decoder,self).__init__()
        self.device = device
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.encoder = Encoder_bones(input_size,hidden_size,num_layers)
        self.encoder_firstPose = Encoder_firstPose(output_size,hidden_size,num_layers,device)
        self.decoder =Decoder(input_size,hidden_size,num_layers,output_size)
        # self.dropout = nn.Dropout(0.0)
          
    def forward(self,input_bone,first_frame_pose): #rest pose is like [frames,vertices] = [43,83550] input_bones = [43,624]       
        batch_size,seq_len,_ = input_bone.size()
        self.batch_size = batch_size
        # print(input_bone.shape,"input")
        # print(first_frame_pose.shape,"first_frame_pose")
        encoder_output_bones,(_,_) = self.encoder(input_bone)
        # print(encoder_output_bones.shape,"encoder_bones")
        encoder_output_first_frame = self.encoder_firstPose(first_frame_pose)
        # print(encoder_output_first_frame.shape,"encoder_rest_pose")
        encoder_output_first_frame = encoder_output_first_frame.unsqueeze(1).expand(-1,seq_len,-1)
        # final_hidden_first_pose = hidden_first_pose.unsqueeze(1).expand(-1,batch_size, -1)
        # final_hidden_bones = hidden_bones
        combine_features = torch.add(encoder_output_first_frame,encoder_output_bones) #να το κάνω συνένωση
        # print(combine_features.shape,"combination")
        # print(combine_features.shape,"this is the shape of the addition")
        # combine_features1 = torch.cat((encoder_output_bones,encoder_output_first_frame),dim=-1)
        # print(combine_features1.shape,"this is the combination with torch cat")
        # print(decoder_input.shape)
        output = self.decoder(combine_features)
        # print(output.shape,"output")
        # output = self.dropout(output)
    
        return output
    
    def set_evaluation_mode(self):
        self.encoder.eval()
        self.encoder_firstPose.eval()
        self.decoder.eval()

    def count_parameters(self):
            return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def print(self):
         print(self)

    #------------------------Dist Per function ------------------------------->
    def DistPer(self,A_orig,A_approx,Avg_mesh_metrices): 
        
        Avg_mesh_metrices = Avg_mesh_metrices.to(self.device)
        Avg_mesh_metrices = Avg_mesh_metrices[0,:]
        # print(Avg_mesh_metrices.shape)

        #reshape because of they are vertices and we must analyze it as vertices (frames,mesh_values,xyz)
        A_orig = A_orig.view(A_orig.shape[0],A_orig.shape[-1]//3,3)   
        A_approx = A_approx.view(A_approx.shape[0],A_approx.shape[-1]//3,3)

        #calculations
        diff_Aorig_Aapprox = torch.norm(A_orig - A_approx, p="fro")
        diff_Aorig_Aavg = torch.norm(A_orig - Avg_mesh_metrices, p="fro")
        disPervalue =  100*(diff_Aorig_Aapprox / (diff_Aorig_Aavg))

        return disPervalue
    
    def train_phase(self, train_loader, criterion ,optimizer):
        
        total_loss = 0.0
        total_DistPer_loss = 0.0
        for batch_idx,(bone_matrices, first_frame, target_mesh, mesh_avg) in enumerate(train_loader):
            bone_matrices = bone_matrices.to(self.device)
            first_frame = first_frame.to(self.device)
            target_mesh =  target_mesh.to(self.device)
            mesh_avg = mesh_avg.to(self.device)

            predicted_mesh =  self.forward(bone_matrices,first_frame)
            predicted_mesh = predicted_mesh.squeeze(0)
            target_mesh = target_mesh.squeeze(0)

            loss = criterion(predicted_mesh,target_mesh)
            if(self.batch_size > 1 ):
                # print(self.batch_size)
                distPer = self.DistPer(target_mesh[self.batch_size-1],predicted_mesh[self.batch_size-1],mesh_avg)
            else:
                distPer = self.DistPer(target_mesh,predicted_mesh,mesh_avg)

            #backpropagation 
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            total_DistPer_loss += distPer.item()            
            
        epoch_loss_disper = total_DistPer_loss / len(train_loader)        
        epoch_loss = total_loss / len(train_loader)  
        return epoch_loss , epoch_loss_disper , predicted_mesh 

    def evaluation_phase(self,evaluate_loader,criterion,mode):
        self.set_evaluation_mode()
        total_loss = 0.0
        total_loss_distPer = 0.0

        for name, param in self.named_parameters():
            if not param.requires_grad:
                print(f"Warning: Parameter {name} is frozen during evaluation.")

        with torch.no_grad():
            for batch_idx,(bone_matrices, first_frame, target_valuation_mesh, mesh_avg) in enumerate(evaluate_loader):

                bone_matrices = bone_matrices.to(self.device)
                first_frame = first_frame.to(self.device)
                target_valuation_mesh = target_valuation_mesh.to(self.device)
                mesh_avg = mesh_avg.to(self.device)

                seq_len = bone_matrices.shape[1]
                
                if (mode=="taketime"):
                    start_time = time.time()

                predicted_valuation_mesh = self.forward(bone_matrices,first_frame)

                if (mode=="taketime"):
                    end_time = time.time()
                    elapse_time_animation = end_time - start_time
                    elapse_time_frame = elapse_time_animation/seq_len
                    print(f"Time for Mesh generation for hole animation: {elapse_time_animation} seconds")
                    print(f"Time for Mesh generation per frame: {elapse_time_frame} seconds")

                predicted_valuation_mesh = predicted_valuation_mesh.squeeze(0)
                target_valuation_mesh = target_valuation_mesh.squeeze(0)

                loss = criterion(predicted_valuation_mesh,target_valuation_mesh)
                distPer = self.DistPer(target_valuation_mesh,predicted_valuation_mesh,mesh_avg)

                total_loss += loss.item()
                total_loss_distPer += distPer.item()

            avg_loss = total_loss / len(evaluate_loader)
            avg_dist_per_loss =total_loss_distPer / len(evaluate_loader)

            return avg_loss , avg_dist_per_loss , predicted_valuation_mesh