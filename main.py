from basic_inpainting import BasicInpainting, import_params
import os
import cv2


def main():
    params = import_params('best_trial.json')
    model = BasicInpainting(**params)

    weights_path = 'trained_model.pth'
    if os.path.exists(weights_path):
        try:
            model.load_pretrained_weights(weights_path)
            print(f'Loaded weights from {weights_path}')
        except Exception as e:
            print(f'Warning: failed to load weights: {e}')
    else:
        print(f'Weights not found at {weights_path}, running with random-initialized model')

    image = 'predicted/images/corrupted_0.png'
    mask = 'predicted/masks/mask0.png'

    output = model.predict(image, mask=mask, show=False)
    out_img = output if not isinstance(output, tuple) else output[0]

    save_dir = 'output/outputs'
    os.makedirs(save_dir, exist_ok=True)
    out_path = os.path.join(save_dir, 'pred_corrupted_0.png')
    cv2.imwrite(out_path, out_img)
    print(f'Prediction saved to {out_path}')


if __name__ == '__main__':
    main()