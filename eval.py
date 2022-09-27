import os.path as osp
import torch
from tqdm import tqdm

from base_container import BaseContainer


class NetworkTester(BaseContainer):
    def __init__(self):
        super().__init__()
        self.init_evaluation_container()

    # main function for validation
    def evaluation(self):
        print('\nEvaluating...')
        self.model.eval()

        dataset_test = self.args.evaluation.dataset.dataset_test
        with open(self.args.evaluation.save_result + '/results.txt', 'w') as f:
            ap_per_id = []
            for idx_dataset in range(len(dataset_test)):
                self.evaluator.reset()
                for i, samples in enumerate(tqdm(self.test_loader[idx_dataset])):
                    samples = to_cuda(samples)

                    with torch.no_grad():
                        outputs = self.model(samples)
                        self.evaluator.add_batch(outputs['feat'], outputs['target'])

                metrics, ap_per_id_ = self.evaluator.run(k_list=self.args.evaluation.recall_k)
                ap_per_id.append(ap_per_id_)
                keys = sorted(list(metrics.keys()))
                f.write('\n' + dataset_test[idx_dataset] + ':\n')
                print('\n' + dataset_test[idx_dataset])
                for k in keys:
                    f.write('%s %.4f\n'%(k, metrics[k]))
                    print('%s %.4f'%(k, metrics[k]))

            torch.save(ap_per_id, osp.join(self.args.evaluation.save_result, 'ap_per_id.pth'))


def to_cuda(sample):
    if isinstance(sample, list):
        return [to_cuda(i) for i in sample]
    elif isinstance(sample, dict):
        for key in sample.keys():
            sample[key] = to_cuda(sample[key])
        return sample
    elif isinstance(sample, torch.Tensor):
        return sample.cuda()
    else:
        return sample

def main():
    tester = NetworkTester()
    tester.evaluation()

if __name__ == "__main__":
    main()
