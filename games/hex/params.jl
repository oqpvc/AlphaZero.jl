Network = NetLib.ResNet

netparams = NetLib.ResNetHP(
    num_blocks = 10,
    num_filters = 64,
    conv_kernel_size = (3, 3))

self_play = SelfPlayParams(
    num_games=2500,
    reset_mcts_every=100,
    use_gpu=true,
    num_workers=256,
    mcts = MctsParams(
        num_iters_per_turn=400,
        cpuct=4.0,
        temperature=PLSchedule([0, 20, 30], [1., 1., 0.3]),
        dirichlet_noise_ϵ=0.25,
        dirichlet_noise_α=0.1))

arena = ArenaParams(
    num_games=100,
    reset_mcts_every=50,
    update_threshold=0.1,
    flip_probability=0.5,
    num_workers=256,
    mcts = MctsParams(
        self_play.mcts,
        temperature=ConstSchedule(0.3),
        dirichlet_noise_ϵ=0.1))

learning = LearningParams(
    use_gpu=true,
    samples_weighing_policy=LOG_WEIGHT,
    l2_regularization=1e-4,
    batch_size=256,
    loss_computation_batch_size=2048,
    optimiser=Adam(lr=2e-3),
    nonvalidity_penalty=1.,
    min_checkpoints_per_epoch=0,
    max_batches_per_checkpoint=5_000,
    num_checkpoints=1)

params = Params(
    arena=arena,
    self_play=self_play,
    learning=learning,
    num_iters=10,
    mem_buffer_size=PLSchedule([0, 20], [200_000, 2_000_000]))

benchmark = [
    Benchmark.Duel(
        Benchmark.Full(self_play.mcts),
        Benchmark.MctsRollouts(self_play.mcts),
        num_games=100,
        num_workers=256,
        use_gpu=true,
        flip_probability=0.5)
]
