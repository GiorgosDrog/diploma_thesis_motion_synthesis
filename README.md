This repository includes all the code related to the work of my Diploma thesis: <b>Motion synthesis about articulated 
and non-articulated characters</b>, during the last year of my studies at the Computer Science and Enginneering Department of University of Ioannina.

<H2>Blender</H2> 

We use the Blender enviroments in which we import the motions from the isotope Mixamo. \n
<u>Examples of motions</u>  

![Screenshot (1472)](https://github.com/GiorgosDrog/diploma_thesis_motion_synthesis/assets/72260809/73cf04e9-8390-491a-8c2c-ab03eece8074)

From motions like this we extract files with numpy format
We already have extracted files in the folder npy_files. These files are used for Datasets about model's training and validation datasets 

For the DataSets we follow the class stucture which are appeared in the schema below.


<H2> WorkFlow </H2>

![ροήδεδομέωνΔιπλωματικη (3)](https://github.com/GiorgosDrog/diploma_thesis_motion_synthesis/assets/72260809/08ea37ae-2d0c-4356-9044-93d4d4c37b46)

<H2> Models architectures </H2>
The models follow an Encoder Decoder architecture. The goal is the extraction meaningful information from bones matrices <br> and with them we want to genereate the mesh vertices of the motion for each animation. for the project have beed developed two architecture, one with LSTM encoder decoder with attention mechanism and the second one with encoder decoder intermidiate calculations.   
The schemmas below show the both architecture

<H3>LSTM_Encoder_decoder_attention</H3>

![image](https://github.com/GiorgosDrog/diploma_thesis_motion_synthesis/assets/72260809/420ae429-ae62-4c8f-8646-69a85df19ad1)

<H3>LSTM_Encoder_Decoder</H3>

![image](https://github.com/GiorgosDrog/diploma_thesis_motion_synthesis/assets/72260809/982f983c-88c1-4c5e-893a-7226149fd3e8)

<H2>Motion Synthesis</H2>
To acheive the goal of the project we develop two methods. At first we take the already trained models and we use them for the moethod. We swap the decoder elements to blend the trained parameters for one motion with the encoded information of the other motion. With this way the decoder generates a mixed motion. The schema below represents the method architecture.

![image](https://github.com/GiorgosDrog/diploma_thesis_motion_synthesis/assets/72260809/728f3ffe-dbe8-4668-a547-9326808deb6c)

Furthermore the second method splits the motions in seperate categories. The categories are seperated with vector like [0.0,0.0,0.0,1.0,0.0] - > Jump , [0.0,0.0,0.0,0.0,1.0] - > Walk , [1.0,0.0,0.0,0.0,0.0] - > point , [0.0,1.0,0.0,0.0,0.0] - >  rotate , [0.0,0.0,1.0,0.0,0.0] - > turn. When we split the animation in categories we trained a another model like the LSTM_Encoder_Decoder_Αttention with addition of an input to the decoder with will takes the cotegory vector. In the training phase the model must understand the connection and the categories. Also it must generate well the mesg vertices for the motion. With these changes at the inference mode of the model we can feed an category like [1.0,0.0,0.0,0.0,1.0] - > Walk + Point which must gives us the a mix motion of walking and pointing.

The schema below represents the architecture of the category model 
<H3>LSTM_encoder_decoder_attention_categories</H3>

![MainTechnique (1) (1)](https://github.com/GiorgosDrog/diploma_thesis_motion_synthesis/assets/72260809/c80cd8e5-567b-4f3d-ae0d-19f9665b7338)



<H2> HOW TO RUN THE CODE </H2>
