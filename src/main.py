import json

from inference_wrapper import InferenceWrapper

if __name__ == '__main__':
    # init
    with open('config.json', 'r') as f:
        config = json.load(f)
    iw = InferenceWrapper(config)
    # compute
    iw.run()
    # save results
    iw.save_coords()
    metrics_str = iw.save_metrics()
    # stdout
    for string in metrics_str:
        print(string, end='')
