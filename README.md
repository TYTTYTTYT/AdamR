# AdamWR
The Adam with Weight Recovery optimizer

# TL;DR
AdamW decays unused parameters towards zero, which makes the model "forget" the pretrained parameters during finetuning. AdamWR instead tries to recover unused parameters towards pretrained values during finetuning.
