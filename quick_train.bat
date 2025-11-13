@echo off
REM Quick training script for text-conditioned UNet
REM Trains for 5 epochs as a quick test

cd src

echo ========================================
echo Training Text-Conditioned UNet Model
echo ========================================
echo.
echo This will train for 5 epochs as a quick test.
echo For full training, use: python train.py --epochs 50
echo.

python train.py ^
    --esc50-path ../data/ESC-50-master/ESC-50-master ^
    --epochs 5 ^
    --batch-size 4 ^
    --lr 0.001 ^
    --output-dir ../checkpoints ^
    --save-every 5 ^
    --num-workers 0

echo.
echo ========================================
echo Training Complete!
echo ========================================
echo.
echo Model saved to: ../checkpoints/text_conditioned_unet_best.pth
echo.
echo To run inference:
echo python inference.py --audio-in mix.wav --audio-out out.wav --prompt-text "dog bark" --mode keep --time-windows 0 4 --model-path ../checkpoints/text_conditioned_unet_best.pth
echo.

pause
