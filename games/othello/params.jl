Network = NetLib.ResNet

netparams = NetLib.ResNetHP(
    num_filters=64,
    num_blocks=3,
    conv_kernel_size=(3, 3),
    num_policy_head_filters=32,
    num_value_head_filters=32,
    batch_norm_momentum=0.1)

self_play = SelfPlayParams(
    num_games=5000,
    num_workers=128,
    use_gpu=true,
    reset_mcts_every=64,
    mcts=MctsParams(
        num_iters_per_turn=600,
        cpuct=2.0,
        prior_temperature=1.0,
        temperature=PLSchedule([0, 20, 30], [1.0, 1.0, 0.3]),
        dirichlet_noise_ϵ=0.25,
        dirichlet_noise_α=0.15))

arena = ArenaParams(
    num_games=128,
    num_workers=128,
    use_gpu=true,
    reset_mcts_every=32,
    flip_probability=0.5,
    update_threshold=0.05,
    mcts=MctsParams(
        self_play.mcts,
        temperature=ConstSchedule(0.2),
        dirichlet_noise_ϵ=0.05))

learning = LearningParams(
    use_gpu=true,
    use_position_averaging=true,
    samples_weighing_policy=LOG_WEIGHT,
    batch_size=1024,
    loss_computation_batch_size=1024,
    optimiser=Adam(lr=2e-3),
    l2_regularization=1e-4,
    nonvalidity_penalty=1.,
    min_checkpoints_per_epoch=1,
    max_batches_per_checkpoint=2000,
    num_checkpoints=1)

params = Params(
    arena=arena,
    self_play=self_play,
    learning=learning,
    num_iters=20,
    ternary_rewards=true,
    use_symmetries=true,
    memory_analysis=nothing,
    mem_buffer_size=PLSchedule(
        [      0,        40],
        [400_000, 2_000_000]))

mcts_baseline =
    Benchmark.MctsRollouts(
        MctsParams(
            arena.mcts,
            num_iters_per_turn=1000,
            cpuct=1.))

minmax_baseline = Benchmark.MinMaxTS(depth=3, amplify_rewards=true, τ=0.2)

benchmark = [
    Benchmark.Duel(
        Benchmark.Full(arena.mcts),
        baseline,
        num_games=128,
        num_workers=128,
        use_gpu=true,
        flip_probability=0.5,
        color_policy=CONTENDER_WHITE)
    for baseline in [mcts_baseline, minmax_baseline]
]
