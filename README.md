This project investigates the robustness of pretrained self-supervised learning (SSL) representations for Speech Emotion Recognition (SER). The central research question is:

Do differences in the pretraining objectives of UniSpeech-SAT and WavLM produce different robustness to unseen speakers and acoustic noise when used as frozen feature extractors for speech emotion recognition?

The study compares UniSpeech-SAT Base+ and WavLM Base+ as frozen feature extractors using speech sampled at 16 kHz. Extracted representations are reduced using temporal mean pooling and classified using identical lightweight MLP classifiers to ensure a controlled comparison between the two SSL models. The RAVDESS speech dataset is used with four emotional-expression categories: neutral, happy, sad, and angry. Actor-independent data splits ensure complete separation between training and test speakers, enabling evaluation of generalisation to unseen speakers.

Two primary experiments are conducted. The first evaluates unseen-speaker performance using clean RAVDESS recordings. The second evaluates acoustic-noise robustness by introducing environmental noise from the DEMAND dataset to the same test recordings under controlled clean, 10 dB SNR, and 0 dB SNR conditions. Performance is assessed using macro-F1, per-emotion F1, confusion matrices, and degradation under increasing noise.

An MFCC → MLP system is retained as a conventional reference baseline. The primary comparison remains the robustness characteristics of the frozen UniSpeech-SAT and WavLM representations.

Completed so far:

A single RAVDESS recording was resampled from 48 kHz to 16 kHz and processed using the frozen UniSpeech-SAT Base+ and WavLM Base+ models.

| Model | Frame-Level Representation | Mean-Pooled Embedding | Processing Time |
|---|---|---|---|
| UniSpeech-SAT Base+ | 164 × 768 | 768 | 1.058 s |
| WavLM Base+ | 164 × 768 | 768 | 0.992 s |

Both models produced 768-dimensional mean-pooled embeddings from the same input recording. Processing times refer to the model forward pass only and exclude model loading and downloading.
