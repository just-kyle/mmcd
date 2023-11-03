import os
import pdb
import onnx
import torch
import numpy as np
import onnxruntime as ort
import torchaudio.compliance.kaldi as ta_kaldi
import sys
sys.path.append('..')
from BEATs_gj import Beats_xhs

def load_model(ckp_path,beats_ckp_path):
    ckp = torch.load(ckp_path, map_location=device)
    new_state_dict = {}
    for key in ckp['state_dict']:
        new_state_dict[key.replace('module.','')] = ckp['state_dict'][key]

    model = Beats_xhs(beats_ckp_path).to(device)
    model.load_state_dict(new_state_dict)
    model.eval()
    return model

def run_onnx(input_data):
    session = ort.InferenceSession(onnx_export_path)
    # 准备输入数据
    input_data2_numpy = np.expand_dims(input_data.cpu().numpy().astype(np.float32)[0,:,:], axis=0)  # 将input_data1转换为Numpy数组
    print(input_data2_numpy.shape)
    # input_data2_numpy = input_data2.cpu().numpy()
    input_name = input_names[0]  # 获取输入名称。为我们之前指定的 input_name

    # 执行推理
    output = session.run(None, {input_name: input_data2_numpy})

    # 输出结果
    print("Onnx Output:", output)

def preprocess(
        source: torch.Tensor,
        fbank_mean: float = 15.41663,
        fbank_std: float = 6.55582,
) -> torch.Tensor:
    fbanks = []
    for waveform in source:
        waveform = waveform.unsqueeze(0) * 2 ** 15
        fbank = ta_kaldi.fbank(waveform, num_mel_bins=128, sample_frequency=16000, frame_length=25, frame_shift=10)
        fbanks.append(fbank)
    fbank = torch.stack(fbanks, dim=0)
    fbank = (fbank - fbank_mean) / (2 * fbank_std)
    return fbank


if __name__ == '__main__':
    model_name = 'human_tts_effects'
    ckp_path = 'checkpoint_0039.pth.tar'
    npy_root = '/data/chaowei/guojie/note_cost/yinxiao/data/audio_segs_8s_05_pad'
    note_id = '61ca9eae000000002103a3b9'  # 61ca9eae000000002103a3b9 61ca9eae000000002103a3b9
    beats_ckp_path = '/mnt/public/usr/guojie/data/projects/note_cost/train_beats/pretrained_models/BEATs_iter3_plus_AS20K.pt'

    onnx_export_path = './{}.onnx'.format(model_name)
    device = torch.device("cuda:7" if torch.cuda.is_available() else "cpu")

    processed_tensor_list = []
    my_list = []
    this_audio_seg_root = os.path.join(npy_root, note_id)
    # TODO 按照时序顺序读取文件
    audio_seg_files = os.listdir(this_audio_seg_root)
    # for audio_seg_file in audio_seg_files:
    this_seg_label_list = []
    for i in range(len(audio_seg_files)):
        audio_seg_file = '{}_{}.npy'.format(i * 8, (i + 1) * 8)

        npy_audio = np.load(os.path.join(this_audio_seg_root, audio_seg_file))
        this_audio_input_16khz = torch.tensor(np.expand_dims(npy_audio, 0)).to(device)
        fbank = preprocess(this_audio_input_16khz, fbank_mean=15.41663, fbank_std=6.55582)
        processed_tensor_list.append(fbank)
        # gt_list_seg_level.append(seg_level_label_dict[os.path.join(this_audio_seg_root, audio_seg_file)])
        # this_seg_label_list.append(seg_level_label_dict[os.path.join(this_audio_seg_root, audio_seg_file)])
    audio_input_16khz = torch.cat(processed_tensor_list, dim=0).to(device)
    seg_num = audio_input_16khz.shape[0]

    model = load_model(ckp_path, beats_ckp_path)
    input_names = [f"{model_name}_input"]
    output_names = [f"{model_name}_output"]

    dynamic_axes = {
    input_names[0]: {0: "seg_num", 1: "seg_length"},
    output_names[0]: {0: "seg_num"}
    }
    with torch.no_grad():
        output = model(audio_input_16khz).detach().cpu().numpy()
    print('Pytorch Output:', output)


    
    torch.onnx.export(
                    model,  # model being run
                    audio_input_16khz,               # model input (or a tuple for multiple inputs)
                    onnx_export_path,            # where to save the onnx model
                    dynamic_axes=dynamic_axes,
                    export_params=True,        # store the trained parameter weights inside the model file
                    opset_version=11,          # the ONNX version to export the model to
                    input_names=input_names,   # the model's input names
                    output_names=output_names  # the model's output names
                    )
    run_onnx(input_data=audio_input_16khz)







