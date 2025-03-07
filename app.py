import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
import os
import numpy as np
import pandas as pd
import collections
import cv2
import ast
# from datasets import load_dataset
import gradio as gr
from collections import defaultdict
from epf_model import EPT
import torch
import json
video_dir = 'Test_videos'  # Directory for videos
file_path = 'FileList.csv'  # Path to CSV files
volume_tracing_path = 'VolumeTracings.csv'  # Path to volume tracing CSV

# Assuming the JSON files are stored in this directory
json_dir = 'Interfaced_results/'  # Update with the actual path

class DataFramePreProcess():
    
    def __init__(self,file_path, volume_tracing_path):
        self.file_path = file_path
        self.volume_tracing_path = volume_tracing_path
        self.video_dir = video_dir
        self.data = pd.read_csv(self.file_path)
    def get_df(self):
        self.data['FileName']=self.data['FileName']+'.avi'
        self.frames_tracing = collections.defaultdict(list)
        with open(self.volume_tracing_path) as f:
            header = f.readline().strip().split(",")
            for line in f:
                filename, _, _, _, _, frame = line.strip().split(',')
                frame = int(frame)
                if frame not in self.frames_tracing[filename]:
                    self.frames_tracing[filename].append(frame)
        ed_es=[]
        for i, f in enumerate(self.data['FileName']):
            ed_es.append(self.frames_tracing[f])
        self.data['ED_ES']=ed_es
        print("-----Complete Data-----")
        print("Total Number of Videos in dataset", len(self.data))
        print("Traing data:", len(self.data[self.data['Split']=='TRAIN']))
        print("Validation data:", len(self.data[self.data['Split']=='VAL']))
        print("Test data:", len(self.data[self.data['Split']=='TEST']))
        print(' ')
        return self.data
    def get_df_with_50_FPS(self):
        self.data = self.get_df()
        self.data=self.data[self.data['FPS']==50]
        print("----Data Frame with FPS ==50----")
        print("Total Number of Videos in dataset", len(self.data))
        print("Traing data:", len(self.data[self.data['Split']=='TRAIN']))
        print("Validation data:", len(self.data[self.data['Split']=='VAL']))
        print("Test data:", len(self.data[self.data['Split']=='TEST']))
        print(' ')
        return self.data
    def get_df_NF_GT_32(self):
        self.data = self.get_df()
        self.data=self.data[self.data['NumberOfFrames']>32]
        self.data.reset_index(inplace=True)
        print("----Data Frame with NF>32----")
        print("Total Number of Videos in dataset", len(self.data))
        print("Traing data:", len(self.data[self.data['Split']=='TRAIN']))
        print("Validation data:", len(self.data[self.data['Split']=='VAL']))
        print("Test data:", len(self.data[self.data['Split']=='TEST']))
        print(' ')
        return self.data
    def get_preprocessed_df(self):
        self.data = self.get_df_NF_GT_32()  # Select any of above preprocessing steps
        for i, ed_es in enumerate(self.data['ED_ES']):
            if len(ed_es)!=2 or ed_es[1]-ed_es[0]<8 or (ed_es[1]-ed_es[0])>20:
                self.data.drop(i,inplace=True)
        list_ed_es=[]
        for ed_es in self.data['ED_ES']:
            list_ed_es.append(str(ed_es))
        self.data['ED_ES']= list_ed_es
        print("-----PreProcessed DataFrame----")
        print("Total Number of Videos in dataset", len(self.data))
        print("Traing data:", len(self.data[self.data['Split']=='TRAIN']))
        print("Validation data:", len(self.data[self.data['Split']=='VAL']))
        print("Test data:", len(self.data[self.data['Split']=='TEST']))
        print(' ')
        
        return self.data
    def show_data(self):
        print(self.data)
if __name__ == '__main__':     
    DataFrame=DataFramePreProcess(file_path,volume_tracing_path)
    DataFrame=DataFrame.get_preprocessed_df()


video_paths = sorted(
    [
        os.path.join(video_dir, fname)
        for fname in os.listdir(video_dir)
        if fname.endswith(".avi")
    ]
)
paths_dict = {path.split('/')[-1]: path for path in video_paths}
DataFrame['VideoPath'] = DataFrame['FileName'].map(paths_dict)
df=DataFrame[["FileName","VideoPath","EF","ESV","EDV","ED_ES","Split"]]
test_df = df[df['Split']=='TEST'][:100]

# Frame extractor class
class FrameExtractor:
    def __init__(self, test_df):
        self.video_paths = video_paths  # List of video paths
        self.test_df = test_df            # DataFrame containing metadata
        
    @staticmethod
    def croped_index(idx, ed_es, max_seq_length=32):
        ed, es = ed_es[0], ed_es[1]
        center = ed + (es - ed) // 2
        if center >= (max_seq_length / 2) and center <= (len(idx) - (max_seq_length / 2)):
            nstart = int(center - (max_seq_length / 2))
            nend = int(center + (max_seq_length / 2))
            n_idx = idx[nstart:nend]
            n_ed = ed - nstart 
            n_es = es - nstart
        elif center < (max_seq_length / 2):
            n_idx = idx[:max_seq_length]
            n_ed = ed
            n_es = es
        elif center > (len(idx) - (max_seq_length / 2)):
            nstart = len(idx) - max_seq_length
            n_idx = idx[nstart:]
            n_ed = ed - nstart
            n_es = es - nstart
            
        return n_idx,(n_ed,n_es)

    @staticmethod
    def get_label(video1, ed_es): 
        label1 = np.zeros(video1.shape[0])
        ed, es = ed_es[0], ed_es[1]
        diff = es - ed
        vlength = video1.shape[0]
        start = ed % diff
        end = start + (vlength // diff -1) * diff+1 
        video = video1[start:end]

        label = np.zeros(video.shape[0])
        
        if ed // diff % 2 == 0:     # If Starting frame is ED
            for n in range(video.shape[0] // diff): # n number of cycles
                for t in range(n * diff, (n + 1) * diff +1 ):  # for nth cycle
                    if n % 2 == 0:  # if ED Phase
                        label[t] = np.power(((n + 1) * diff - t) / diff, 1)
                    else:  # if ES Phase
                        label[t] = np.power((t - n * diff) / diff, 1)
        else:
            for n in range(video.shape[0] // diff):
                for t in range(n * diff, (n + 1) * diff +1):
                    if n % 2 == 0:
                        label[t] = np.power((t - n * diff) / diff, 1)
                    else:
                        label[t] = np.power(((n + 1) * diff - t) / diff, 1) 
        
        label1[start:end] = label
        last_values = label1.shape[0] - label.shape[0] - start+1
        
        if label[0]!=label[-1]:
            label1[end:]=np.flip(label)[:last_values][1:]
        else:
            label1[end:] = label[:last_values][1:]
        if start != 0.:
            label1[:start] = label[:2 * diff][-start:]
        
        return label1

    def video_frame_with_labels(self, video_index):
        if video_index < 0 or video_index >= len(self.test_df['VideoPath'].values):
            return [], f"Invalid index. Please select an index between 0 and {len(self.video_paths) - 1}"

        video_path = self.test_df['VideoPath'].values[video_index]
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frames = []

        for i in range(total_frames):
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)

        cap.release()
        frames = np.stack(frames)
        ed_es = ast.literal_eval(self.test_df['ED_ES'].values[video_index])
        label = self.get_label(frames, ed_es)
        v_index = np.arange(frames.shape[0])
        new_index,ed_es1 = self.croped_index(v_index, ed_es, max_seq_length=32)
        
        frames = frames[new_index]
        label = label[new_index]
        
        return torch.tensor(frames[:, :, :, 0]),label,ed_es1
        
def build_model():
    model = EPT(
            in_chans=32,
            num_classes=32,
            embed_dims=[32, 80, 160, 256],
            depths=[2, 2, 6, 2],
            kernel_sizes=[3, 5, 7, 9],
            num_heads=[2, 5, 10, 16],
            window_sizes=[8, 4, 2, 1],
            mlp_kernel_sizes=[5, 5, 5, 5],
            mlp_ratios=[4, 4, 4, 4],
            drop_path_rate=0.1,
            # qkv_bias=True,
            # norm_layer=torch.nn.LayerNorm,
            # pretrained=False
            )
    
    model.load_state_dict(torch.load('model', map_location='cpu', weights_only=True))
    return model




# Function to load all JSON files and extract user information and selected frames
def load_json_data(json_dir):
    json_files = [f for f in os.listdir(json_dir) if f.endswith('.json')]
    all_data = []
    
    for file in json_files:
        with open(os.path.join(json_dir, file), 'r') as f:
            for line in f:
                data = json.loads(line.strip())
                if 'user_info' in data:
                    user_info = data['user_info']
                elif 'video_index' in data:
                    video_info = data
                    all_data.append({'user_info': user_info, 'video_info': video_info})
    
    return all_data

# Your original loaded data
loaded_data = load_json_data(json_dir)

# Using defaultdict to group video info by user info
grouped_data = defaultdict(lambda: {'user_info': None, 'videos': []})

for entry in loaded_data:
    user_info = tuple(entry['user_info'].items())  # Convert user_info dict to tuple (so it can be used as a key)
    if not grouped_data[user_info]['user_info']:
        grouped_data[user_info]['user_info'] = entry['user_info']
    
    grouped_data[user_info]['videos'].append(entry['video_info'])

# Convert defaultdict to a regular list of dicts
cleaned_data = list(grouped_data.values())   

def EPD_result_plot(x_test, y_test, y_pred, t_ed, t_es, p_ed, p_es, R2, i):
    # Create a figure and subplots
    fig, axs = plt.subplots(1, 2, figsize=(10, 5), gridspec_kw={'width_ratios': [2, 2]})

    # Plot on the left half
    axs[0].plot(y_test)
    axs[0].plot(y_pred)
    axs[0].legend(['Ground Truth', 'Predicted'], fontsize=12)

    # Adjust the spacing between the plots
    plt.subplots_adjust(left=0.05, right=0.95, top=0.85, bottom=0.1, wspace=0.3)

    # Create a grid for the right half
    axs_right = fig.add_subplot(axs[1])
    axs_right.set_xticks([])
    axs_right.set_yticks([])

    # Plot images on the right half, maintaining aspect ratio and adjusting insets
    axs_right_top_left = axs_right.inset_axes([0, 0.5, 0.48, 0.48])  # Reduced width to remove gap
    axs_right_top_left.imshow(x_test[t_ed].cpu(), aspect='auto')
    axs_right_top_left.set_title('T_ED', fontsize=12)
    axs_right_top_left.set_xticks([])
    axs_right_top_left.set_yticks([])

    axs_right_top_right = axs_right.inset_axes([0.5, 0.5, 0.48, 0.48])  # Reduced width to remove gap
    axs_right_top_right.imshow(x_test[p_ed].cpu(), aspect='auto')
    axs_right_top_right.set_title('P_ED', fontsize=12)
    axs_right_top_right.set_xticks([])
    axs_right_top_right.set_yticks([])

    axs_right_bottom_left = axs_right.inset_axes([0, 0, 0.48, 0.48])  # Ensure tight fit between rows
    axs_right_bottom_left.imshow(x_test[t_es].cpu(), aspect='auto')
    axs_right_bottom_left.set_xlabel('T_ES', fontsize=12)
    axs_right_bottom_left.set_xticks([])
    axs_right_bottom_left.set_yticks([])

    axs_right_bottom_right = axs_right.inset_axes([0.5, 0, 0.48, 0.48])  # Ensure tight fit between rows
    axs_right_bottom_right.imshow(x_test[p_es].cpu(), aspect='auto')
    axs_right_bottom_right.set_xlabel('P_ES', fontsize=12)
    axs_right_bottom_right.set_xticks([])
    axs_right_bottom_right.set_yticks([])

    # Set the main title and tighten the layout to maximize space usage
    plt.suptitle('T_ED: ' + str(t_ed) + ', P_ED: ' + str(p_ed) + ', T_ES: ' + str(t_es) + ', P_ES: ' + str(p_es) + ', R2_score: ' + str(R2)[:6], fontsize=15)
    plt.tight_layout()

    # Save the figure and show it
    # fig.savefig('Echo Phase Detection Output_' + str(i))
    return fig

def get_table_data(video_index):
    """
    Extracts the ED and ES frame data from cleaned_data for the selected video index.
    """
    table_data = []
    for entry in cleaned_data:
        user_name = entry['user_info']['name']
        for video in entry['videos']:
            if video['video_index'] == video_index:
                ed_frame = video['selected_frames'][0]
                es_frame = video['selected_frames'][1]
                table_data.append([user_name, ed_frame, es_frame])
    
    # Convert to pandas DataFrame for cleaner tabular display
    df = pd.DataFrame(table_data, columns=["User Name", "Selected ED Frame", "Selected ES Frame"])
    return df

def visualize_predictions(video_index):
    # Extract frames and labels from FrameExtractor
    frame_extractor = FrameExtractor(test_df)
    frames, labels,ed_es = frame_extractor.video_frame_with_labels(video_index)
    model = build_model()
    y_pred = model(frames.unsqueeze(0)/255).detach().numpy()[0]
    R2=r2_score(labels,y_pred)
    t_ed = ed_es[0]
    t_es = ed_es[1]
    p_ed = np.min(np.where(y_pred==y_pred.max()))
    p_es = np.max(np.where(y_pred==y_pred.min()))
    
    fig = EPD_result_plot(frames, labels, y_pred, t_ed, t_es, p_ed, p_es, R2, video_index)

    # Get table data
    table = get_table_data(video_index)

    return fig, table

def gradio_interface():
    with gr.Blocks() as interface:
        with gr.Column():
            video_index_input = gr.Slider(minimum=0, maximum=len(test_df)-1, value=0, step=1, label='Select Video Index')
            output_plot = gr.Plot(label="Prediction Plot")
            output_table = gr.DataFrame(headers=["User Name", "Selected ED Frame", "Selected ES Frame"], label="ED/ES Frames Table")
        
        # Automatically update the plot and table when the video index changes
        video_index_input.change(fn=visualize_predictions, inputs=video_index_input, outputs=[output_plot, output_table])

    interface.launch()

if __name__ == "__main__":
    gradio_interface()