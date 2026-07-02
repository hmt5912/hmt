
tasks_hip = {
    "2-2": {
            0: [0, 1, 2],  #
            1: [3,4],  # 
            2: [5,6]
    }
}



tasks_amos = {
    "13-2":
        {
            0: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,13],
            1: [14,15],
            2: [16]
        },
    "16-0":
        {
            0: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13,14,15,16],
    },
}


def get_task_list():
    return list(tasks_amos.keys()) + list(tasks_hip.keys())


def get_task_labels(dataset, name, step):
    if dataset == 'amos':
        task_dict = tasks_amos[name]
    elif dataset == 'hip':
        task_dict = tasks_hip[name]
    else:
        raise NotImplementedError
    assert step in task_dict.keys(), f"You should provide a valid step! [{step} is out of range]"

    labels = list(task_dict[step])
    labels_old = [label for s in range(step) for label in task_dict[s]]
    return labels, labels_old, f'data/{dataset}/{name}'


def get_per_task_classes(dataset, name, step):
    if dataset == 'amos':
        task_dict = tasks_amos[name]
    elif dataset == 'hip':
        task_dict = tasks_hip[name]
    else:
        raise NotImplementedError
    assert step in task_dict.keys(), f"You should provide a valid step! [{step} is out of range]"

    classes = [len(task_dict[s]) for s in range(step + 1)]
    return classes