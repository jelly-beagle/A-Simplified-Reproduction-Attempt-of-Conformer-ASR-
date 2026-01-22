# A-Simplified-Reproduction-Attempt-of-Conformer-ASR-
This project is aiming at validating the core hybrid modeling on Mandarin Chinese ASR tasks within a constrained training notebook on Kaggle,which based on reproduction for conformer architecture
# model_layer and subsample
This project reproduces and verifies the feasibility of the RepVGG-SE-Conformer model structure proposed in the paper "End-to-End Method Based on Conformer for Speech Recognition".
Compared to the standard Conformer, this model uses a multi-branch RepVGG training network at the acoustic input stage to reduce computational complexity and capture diverse features. It integrates SEBlock to apply squeeze-and-excitation mechanisms for channel feature compensation and implements a switch_to_deploy function for the conversion from training to inference.
本项目简单复现并验证了论文《基于Conformer的端到端语音识别方法》（end-to-end method based on Conformer for speech recognition）中提出的RepVGG-SE-Conformer 这一模型结构的可行性。该模型针对经典的conformer结构，在编码声学信息输入端使用了多分支结构的RepVGG训练网络以降低计算复杂度、捕捉多种特征，使用SEBlock注意力结构引入压缩和奖励机制，弥补缺失的通道特征，实现了从训练转化为推理的switch_to_deploy 函数。
# the paper
[《基于Conformer的端到端语音识别方法》](https://www.arocmag.cn/abs/2023.11.0563)
![Rep-VGG-Conformer](.)<img width="805" height="569" alt="屏幕截图 2026-01-22 201043" src="https://github.com/user-attachments/assets/080f9a45-22d9-45c6-931b-fa20bed1af42" />
# extracted data
[aishell](https://www.kaggle.com/datasets/seriousken/aishell-1)
# simplified experiment
This project is positioned as an architectural prototype validation. Due to limited computational resources, the core Conformer structure has been simplified by reducing the FFN to a single layer, utilizing PyTorch's built-in absolute positional encoding, compressing the encoder to 6 layers, and setting the number of attention heads to 4. At the decoding stage, a single-layer CTC decoder is implemented. The model was trained using audio data from subsets S001–S020 of the AISHELL-1 open-source Mandarin speech corpus, with 10 experimental epochs conducted to verify the runnability of the code. Therefore, for a complete reproduction, all layer counts and dimensions should be restored, and a warmup-enabled optimizer suitable for Transformer architectures should be selected.
本项目定位为架构原型验证，由于计算资源有限，对于conformer的核心结构，简化FFN结构为单层，使用pytorch内置绝对位置编码，压缩encoder层数为6层，多头头数为4；在解码层面，设置CTC单层解码。数据集使用 AISHELL-1 开源中文语音数据集训练集中S001-S020音频数据，进行10轮实验查验代码的可运行性。因此，在完整的复现过程中，应还原各类层数和维度，包括选取适合transformer结构的可预热优化器。
[WeNet 的标准 Conformer 编码器](https://github.com/wenet-e2e/wenet)
