import json
import os
import subprocess
import sys
import click

dataset_configs = {
    "sunrgbd": [
        dict(
            name="fcaf3d",
            config="configs/fcaf3d/fcaf3d_sunrgbd-3d-10class.py",
            checkpoint="/nfs/andrew/bbox/fcaf3d/work_dirs/fcaf3d_sunrgbd-3d-10class/latest.pth",
        ),
        dict(
            name="autoreg",
            config="configs/fcaf3d/autoregcond_fcaf3d_sunrgbd-3d-10class.py",
            checkpoint="/nfs/andrew/bbox/fcaf3d/work_dirs/autoregcond_iou_scratch/epoch_30.pth",
        ),
        dict(
            name="votenet",
            config="configs/votenet/votenet-v2_16x8_sunrgbd-3d-10class.py",
            checkpoint="/nfs/andrew/bbox/fcaf3d/work_dirs/snapshots/votenet_20211016_132950.pth",
        ),
        dict(
            name="imvotenet",
            config="configs/imvotenet/imvotenet-v2_stage2_16x8_sunrgbd-3d-10class.py",
            checkpoint="/nfs/andrew/bbox/fcaf3d/work_dirs/snapshots/imvotenet_20211009_131500.pth",
        ),
        # dict(
        #     name="autoreg2",
        #     config="configs/fcaf3d/autoregcond_fcaf3d_sunrgbd-3d-10class.py",
        #     checkpoint="/nfs/andrew/bbox/fcaf3d/work_dirs/autoregcond_iou2/epoch_30.pth",
        # ),
        # dict(
        #     name="autoreg3",
        #     config="configs/fcaf3d/autoregcond_fcaf3d_sunrgbd-3d-10class.py",
        #     checkpoint="/nfs/andrew/bbox/fcaf3d/work_dirs/autoregcond_iou3/epoch_30.pth",
        # ),
        # dict(
        #     name="autoreg4",
        #     config="configs/fcaf3d/autoregcond_fcaf3d_sunrgbd-3d-10class.py",
        #     checkpoint="/nfs/andrew/bbox/fcaf3d/work_dirs/autoregcond_iou4/epoch_30.pth",
        # ),
        # dict(
        #     name="autoreg5",
        #     config="configs/fcaf3d/autoregcond_fcaf3d_sunrgbd-3d-10class.py",
        #     checkpoint="/nfs/andrew/bbox/fcaf3d/work_dirs/autoregcond_iou5/epoch_31.pth",
        # ),
    ],
    "scannet": [
        dict(
            name="fcaf3d",
            config="configs/fcaf3d/fcaf3d_scannet-3d-18class.py",
            checkpoint="/nfs/andrew/bbox/fcaf3d/work_dirs/fcaf3d_scannet_from_snapshot/epoch_12.pth",
        ),
        dict(
            name="autoreg",
            config="configs/fcaf3d/autoregcond_fcaf3d_scannet-3d-18class.py",
            checkpoint="/nfs/andrew/bbox/fcaf3d/work_dirs/autoreg_fcaf3d_scannet_from_snapshot/epoch_12.pth",
        )
    ],
}

@click.command()
@click.argument("dataset")
@click.argument("run_name")
@click.option("--recompute", is_flag=True)
def main(dataset, run_name, recompute):
    benchmark_results = {}
    for model in dataset_configs[dataset]:
        main_dir = f"/nfs/andrew/bbox/benchmark/{run_name}/{dataset}/"
        os.makedirs(main_dir, exist_ok=True)
        if model['name'] == "fcaf3d":
            configs = [
                dict(name="fcaf3d", args=[]),
                ]
        elif "autoreg" in model['name']:
            configs = [
                dict(name="beam", args=["--cfg-options", "model.test_cfg.mode=beam"]),
                ]
            for quantile in [0.1, 0.2, 0.3, 0.4, 0.45, 0.5, 0.55, 0.6]:#, 0.7, 0.8, 0.9]:
            # for quantile in [0.45]:
                configs.append(
                    dict(name=f"quantile_{quantile}", 
                    args=["--cfg-options", "model.test_cfg.mode=quantile", f"model.test_cfg.quantile={quantile}"])
                )
        else:
            configs = [
                dict(name=model['name'], args=[]),
                ]
            # raise ValueError(model['name'])
        for config in configs:
            variant = f"{model['name']}_{config['name']}"
            print(f"variant {variant}")
            save_dir = os.path.join(main_dir, variant)
            os.makedirs(save_dir, exist_ok=True)
            save_results = os.path.join(save_dir, "results.json")

            if os.path.exists(save_results):
                benchmark_results[variant] = save_results
                print(f"skipping {variant}")
                continue

            command = ["tools/dist_test.sh"]
            command += [model['config'], model['checkpoint'], "8"]
            command += ['--eval', 'mAP']
            command += ['--out', save_results]
            command += config['args']

            print(" ".join(command))
            subprocess.check_call(command)

            print(variant, save_results)
            benchmark_results[variant] = save_results

            with open(os.path.join(main_dir, "benchmark.json"), 'w') as jfile:
                json.dump(benchmark_results, jfile)

    with open(os.path.join(main_dir, "benchmark.json"), 'w') as jfile:
        json.dump(benchmark_results, jfile)
    print(benchmark_results)


if __name__ == "__main__":
    main()