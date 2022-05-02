import numpy as np
import matplotlib.pyplot as plt

tasks = ["task_2_1", "task_2_2", "task_2_3_0", "task_2_3_1", "task_2_3_2", "task_2_3_3"]
task_labels = ["Task 2.1", "Task 2.2", "Task 2.3.0", "Task 2.3.1", "Task 2.3.2", "Task 2.3.3"]
task_colors = ["#0d00ff", "#77bdf7", "#e0de63", "#e08d1f", "#e01f1f", "#941616"]
params_color = "#035e0f"

params = {
"task_2_1_params": 4190464,
"task_2_2_params": 4190464,
"task_2_3_0_params": 23958598,
"task_2_3_1_params": 27334726,
"task_2_3_2_params": 27334726,
"task_2_3_3_params": 31402062
}

fps = {
"task_2_1_fps": [6.940703992205794, 6.889900228695583, 6.896965537025922, 6.921363285647541, 6.907804474289208, 6.871407424858029, 6.8684960760705644, 6.831446789207598, 6.790941495308745, 6.8809930249604285, 6.847965131134578, 6.792753535077203, 6.81339795120021, 6.910160616804273, 6.874205985550627, 6.843577453981786, 6.94022726488259, 6.535799240160109, 6.654197745631092, 6.574772101667954],
"task_2_2_fps": [6.939671146632681, 6.870311370771349, 6.928710651154131, 7.006182415973874, 6.865660242119587, 6.793770508909018, 6.817598949366141, 6.861140772225132, 6.904214343692032, 6.878396146435655, 6.421310272753172, 6.841300747897011, 6.791464573574785, 6.3645193777298665, 6.663882676883074, 6.931286104751971, 6.747901717966228, 6.584473257633915, 6.767152621322468, 6.823902342034653],
"task_2_3_0_fps": [10.362289691524099, 9.528418367659885, 9.81539226522647, 10.163833039179464, 9.535614667231396, 9.501021840799423, 10.0764188360408, 10.017627740693799, 9.959366639689202, 9.914516539838683, 9.9386320917065, 9.864242211352952, 9.811960183920917, 9.999764209585111, 10.012226730660068, 10.052517360726224, 9.97937721065312, 10.051923986696186, 9.949251583011876, 10.11445165257896],
"task_2_3_1_fps": [9.99095043663887, 9.834661635486516, 9.679288907678878, 9.869401561831367, 9.993373020239236, 9.92545726738931, 10.063744183864188, 10.02074195802956, 10.152804856416727, 10.055609194370572, 10.077477063267937, 10.066163548566102, 10.05077140789144, 10.068440025576882, 10.00469260973242, 10.047271693916459, 10.094429017902257, 10.042799481499639, 9.951711597618207, 10.04519627499196],
"task_2_3_2_fps": [10.016641130142942, 10.255458802181746, 10.008061226379203, 9.844031291611575, 9.973774725927742, 9.998815676777786, 10.051845694501878, 9.981150461957654, 9.755010459362337, 9.77897960409847, 9.82309706453314, 9.936696883344242, 9.861995676837031, 9.779315453700535, 9.861952082945374, 10.013435031327914, 10.01169450160783, 9.879585671634759, 9.841329565336178, 10.044066404522923],
"task_2_3_3_fps": [3.10780163200368, 3.088401227669059, 3.0761674784463677, 3.0859663029546804, 3.080202445255261, 3.073244243748607, 3.0726586561897724, 3.05542295790011, 3.048273263458127, 3.0826402642058977, 3.0862840247165617, 3.0762406909208146, 3.0910063459715453, 3.091408384114426, 3.092805793866884, 3.0934090109931005, 3.084093161480862, 3.0800556011814777, 3.080109704762627, 3.068190627102439]
#task_2_4_1_fps =
}

if __name__ == '__main__':
    plot_dict = {}
    fig, ax_fps = plt.subplots()
    ax_params = ax_fps.twinx()
    ax_params.set_ylabel('Parameters', color="#035e0f")
    for index, task in enumerate(tasks):
        # Create Dict assigning all the Values to the tasks
        plot_dict[task] = {"params": params[task + "_params"],
                            "fps": fps[task + "_fps"],
                            "stats_fps": (np.mean(fps[task + "_fps"]), np.std(fps[task + "_fps"])),
                            "color": task_colors[index]
                            }
        # Plot over FPS values including 2*sigma error bars
        ax_fps.bar(index, plot_dict[task]["stats_fps"][0],
                    yerr=2*plot_dict[task]["stats_fps"][1], align='center',
                    alpha=0.5, ecolor='black', capsize=10,
                    color=plot_dict[task]["color"]
                    )
    # Plot number of parameters on a second axis
    ax_params.plot([index for index, value in enumerate(task_labels)],
                    [plot_dict[task]["params"] for task in tasks],
                    marker=".", color=params_color, linestyle="--", ms=10)
    # Configure parameters plot_dict
    ax_params.tick_params(axis='y', colors=params_color)
    ax_params.spines['right'].set_color(params_color)
    # Configure FPS plot
    ax_fps.set_ylabel('Frames per Second (fps)')
    ax_fps.set_xticks([index for index, value in enumerate(task_labels)])
    ax_fps.set_xticklabels(task_labels, rotation=45, horizontalalignment='right')
    #ax.set_title('FPS')
    ax_fps.yaxis.grid(b=True, which='major', color='gray', linestyle='--')
    ax_fps.yaxis.grid(b=True, which='minor', color='gray', linestyle='--', alpha=0.2)
    ax_fps.tick_params(axis='x', which='minor', bottom=False)
    plt.minorticks_on()
    # Save the figure and show
    plt.tight_layout()
    plt.savefig('performance_assessment/eval_fps.png')
    plt.show()
