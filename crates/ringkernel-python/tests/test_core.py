"""Tests for core ringkernel functionality."""

import pytest


class TestHlc:
    """Tests for Hybrid Logical Clock."""

    def test_timestamp_creation(self):
        import ringkernel

        ts = ringkernel.HlcTimestamp(physical=1000, logical=5, node_id=1)
        assert ts.physical == 1000
        assert ts.logical == 5
        assert ts.node_id == 1

    def test_timestamp_now(self):
        import ringkernel

        ts = ringkernel.HlcTimestamp.now(node_id=42)
        assert ts.node_id == 42
        assert ts.physical > 0

    def test_timestamp_zero(self):
        import ringkernel

        ts = ringkernel.HlcTimestamp.zero()
        assert ts.is_zero()
        assert ts.physical == 0
        assert ts.logical == 0

    def test_timestamp_ordering(self):
        import ringkernel

        ts1 = ringkernel.HlcTimestamp(100, 0, 1)
        ts2 = ringkernel.HlcTimestamp(100, 1, 1)
        ts3 = ringkernel.HlcTimestamp(101, 0, 1)

        assert ts1 < ts2
        assert ts2 < ts3
        assert ts1 < ts3

    def test_timestamp_pack_unpack(self):
        import ringkernel

        ts = ringkernel.HlcTimestamp(12345, 67, 89)
        packed = ts.pack()
        unpacked = ringkernel.HlcTimestamp.unpack(packed)

        assert ts.physical == unpacked.physical
        assert ts.logical == unpacked.logical
        # Note: node_id may be truncated in packing

    def test_clock_tick(self):
        import ringkernel

        clock = ringkernel.HlcClock(node_id=1)
        ts1 = clock.tick()
        ts2 = clock.tick()

        assert ts2 > ts1
        assert clock.node_id == 1


class TestMessage:
    """Tests for message types."""

    def test_message_id_generate(self):
        import ringkernel

        id1 = ringkernel.MessageId.generate()
        id2 = ringkernel.MessageId.generate()

        assert id1.value != id2.value

    def test_message_id_explicit(self):
        import ringkernel

        msg_id = ringkernel.MessageId(12345)
        assert msg_id.value == 12345
        assert int(msg_id) == 12345

    def test_correlation_id(self):
        import ringkernel

        corr = ringkernel.CorrelationId.generate()
        assert corr.is_some()

        none_corr = ringkernel.CorrelationId.none()
        assert not none_corr.is_some()
        assert not bool(none_corr)

    def test_priority(self):
        import ringkernel

        assert ringkernel.Priority.Low.value == 0
        assert ringkernel.Priority.Normal.value == 1
        assert ringkernel.Priority.High.value == 2
        assert ringkernel.Priority.Critical.value == 3


class TestQueue:
    """Tests for queue types."""

    def test_queue_tier_capacity(self):
        import ringkernel

        assert ringkernel.QueueTier.Small.capacity == 256
        assert ringkernel.QueueTier.Medium.capacity == 1024
        assert ringkernel.QueueTier.Large.capacity == 4096

    def test_queue_tier_upgrade(self):
        import ringkernel

        tier = ringkernel.QueueTier.Small
        upgraded = tier.upgrade()
        assert upgraded == ringkernel.QueueTier.Medium

    def test_queue_monitor(self):
        import ringkernel

        monitor = ringkernel.QueueMonitor(warning_threshold=0.75, critical_threshold=0.90)

        # Healthy
        health = monitor.check_health(depth=50, capacity=100)
        assert health == ringkernel.QueueHealth.Healthy

        # Warning
        health = monitor.check_health(depth=80, capacity=100)
        assert health == ringkernel.QueueHealth.Warning

        # Critical
        health = monitor.check_health(depth=95, capacity=100)
        assert health == ringkernel.QueueHealth.Critical


class TestHybrid:
    """Tests for hybrid dispatcher."""

    def test_processing_mode(self):
        import ringkernel

        assert int(ringkernel.ProcessingMode.CpuOnly) == 0
        assert int(ringkernel.ProcessingMode.Adaptive) == 3

    def test_dispatcher_cpu_only(self):
        import ringkernel

        config = ringkernel.HybridConfig.cpu_only()
        dispatcher = ringkernel.HybridDispatcher(config)

        # CPU-only never uses GPU
        assert not dispatcher.should_use_gpu(1000000)

    def test_dispatcher_hybrid(self):
        import ringkernel

        config = ringkernel.HybridConfig(
            mode=ringkernel.ProcessingMode.Hybrid,
            gpu_threshold=1000,
            gpu_available=True,
        )
        dispatcher = ringkernel.HybridDispatcher(config)

        # Below threshold -> CPU
        assert not dispatcher.should_use_gpu(500)

        # Above threshold -> GPU
        assert dispatcher.should_use_gpu(2000)

    def test_dispatcher_stats(self):
        import ringkernel

        config = ringkernel.HybridConfig.cpu_only()
        dispatcher = ringkernel.HybridDispatcher(config)

        dispatcher.record_cpu_execution()
        dispatcher.record_cpu_execution()

        stats = dispatcher.stats()
        assert stats.cpu_count == 2
        assert stats.gpu_count == 0
        assert stats.total() == 2


class TestResource:
    """Tests for resource management."""

    def test_memory_estimate(self):
        import ringkernel

        est = ringkernel.MemoryEstimate(
            primary_bytes=1000,
            auxiliary_bytes=500,
            peak_bytes=2000,
            confidence=0.9,
        )

        assert est.total_bytes() == 1500
        assert est.peak_bytes == 2000
        # Float precision: f32 doesn't exactly represent 0.9
        assert abs(est.confidence - 0.9) < 0.001

    def test_resource_guard(self):
        import ringkernel

        guard = ringkernel.ResourceGuard(max_memory_bytes=10000, safety_margin=0.1)

        # Can allocate within limit
        assert guard.can_allocate(5000)

        # Track allocation
        guard.record_allocation(5000)

        # Now limited
        assert not guard.can_allocate(5000)

        # Release
        guard.record_deallocation(5000)
        assert guard.can_allocate(5000)

    def test_resource_guard_unguarded(self):
        import ringkernel

        guard = ringkernel.ResourceGuard.unguarded()

        # Always allows allocation
        assert guard.can_allocate(2**60)

    def test_reservation_guard(self):
        import ringkernel

        guard = ringkernel.ResourceGuard(max_memory_bytes=10000)

        # Reserve memory
        with guard.reserve(1000) as reservation:
            assert reservation.bytes == 1000
            assert not reservation.is_committed()
            reservation.commit()
            assert reservation.is_committed()


class TestLaunchOptions:
    """Tests for launch options."""

    def test_default_options(self):
        import ringkernel

        opts = ringkernel.LaunchOptions()
        assert opts.block_size == 256
        assert opts.grid_size == 1
        assert opts.auto_activate is True

    def test_builder_pattern(self):
        import ringkernel

        opts = (
            ringkernel.LaunchOptions()
            .with_queue_capacity(2048)
            .with_block_size(512)
            .with_k2k(True)
        )

        assert opts.input_queue_capacity == 2048
        assert opts.output_queue_capacity == 2048
        assert opts.block_size == 512
        assert opts.enable_k2k is True

    def test_single_block(self):
        import ringkernel

        opts = ringkernel.LaunchOptions.single_block(threads=128)
        assert opts.grid_size == 1
        assert opts.block_size == 128


@pytest.mark.asyncio
class TestRuntimeAsync:
    """Async tests for runtime."""

    async def test_create_runtime(self):
        import ringkernel

        runtime = await ringkernel.RingKernel.create(backend="cpu")
        assert runtime.backend == ringkernel.Backend.Cpu
        assert not runtime.is_shutdown

        await runtime.shutdown()
        assert runtime.is_shutdown

    async def test_launch_kernel(self):
        import ringkernel

        runtime = await ringkernel.RingKernel.create(backend="cpu")

        kernel = await runtime.launch("test_kernel")
        assert kernel.id == "test_kernel"
        assert kernel.is_active()

        await kernel.terminate()
        await runtime.shutdown()

    async def test_list_kernels(self):
        import ringkernel

        runtime = await ringkernel.RingKernel.create(backend="cpu")

        await runtime.launch("kernel1")
        await runtime.launch("kernel2")

        kernels = runtime.list_kernels()
        assert len(kernels) == 2
        assert "kernel1" in kernels
        assert "kernel2" in kernels

        await runtime.shutdown()

    async def test_context_manager(self):
        import ringkernel

        # Note: Async context manager support requires proper __aenter__/__aexit__
        # For now, test manual lifecycle management
        runtime = await ringkernel.RingKernel.create()
        try:
            kernel = await runtime.launch("test")
            assert kernel.is_active()
            await kernel.terminate()
        finally:
            await runtime.shutdown()

        assert runtime.is_shutdown


class TestRuntimeSync:
    """Sync tests for runtime."""

    def test_create_sync(self):
        import ringkernel

        runtime = ringkernel.RingKernel.create_sync(backend="cpu")
        assert runtime.backend == ringkernel.Backend.Cpu

        runtime.shutdown_sync()

    def test_launch_sync(self):
        import ringkernel

        runtime = ringkernel.RingKernel.create_sync(backend="cpu")
        kernel = runtime.launch_sync("test")

        assert kernel.id == "test"
        assert kernel.is_active()

        runtime.shutdown_sync()

    def test_sync_context_manager(self):
        import ringkernel

        with ringkernel.RingKernel.create_sync() as runtime:
            kernel = runtime.launch_sync("test")
            assert kernel.is_active()

        assert runtime.is_shutdown


class TestK2K:
    """Tests for K2K messaging."""

    def test_k2k_config(self):
        import ringkernel

        config = ringkernel.K2KConfig(
            max_pending_messages=2048,
            delivery_timeout_ms=10000,
        )

        assert config.max_pending_messages == 2048
        assert config.delivery_timeout_ms == 10000

    def test_k2k_broker(self):
        import ringkernel

        config = ringkernel.K2KConfig.default()
        broker = ringkernel.K2KBroker(config)

        # Broker starts with no registered kernels
        assert len(broker.registered_kernels()) == 0

        # Stats should be available
        stats = broker.stats()
        assert stats.registered_endpoints == 0
        assert stats.messages_delivered == 0

        # Routes can be added
        broker.add_route("kernel1", "kernel2")
        # Note: Registration happens through runtime, not directly on broker

    def test_delivery_status(self):
        import ringkernel

        assert ringkernel.DeliveryStatus.Delivered.is_success()
        assert ringkernel.DeliveryStatus.Pending.is_pending()
        assert ringkernel.DeliveryStatus.QueueFull.is_failure()


class TestBackend:
    """Tests for backend selection."""

    def test_backend_enum(self):
        import ringkernel

        assert int(ringkernel.Backend.Auto) == 0
        assert int(ringkernel.Backend.Cpu) == 1
        assert int(ringkernel.Backend.Cuda) == 2


class TestCuda:
    """Tests for CUDA functionality."""

    def test_cuda_availability_check(self):
        import ringkernel

        # Should not raise
        result = ringkernel.is_cuda_available()
        assert isinstance(result, bool)

    @pytest.mark.skipif(
        not __import__("ringkernel").is_cuda_available(), reason="CUDA not available"
    )
    def test_cuda_device_creation(self):
        import ringkernel

        device = ringkernel.CudaDevice(0)
        assert device.ordinal == 0
        assert len(device.name) > 0

    @pytest.mark.skipif(
        not __import__("ringkernel").is_cuda_available(), reason="CUDA not available"
    )
    def test_cuda_device_info(self):
        import ringkernel

        device = ringkernel.CudaDevice(0)
        cc = device.compute_capability
        assert isinstance(cc, tuple)
        assert len(cc) == 2
        assert cc[0] > 0  # Major version

    @pytest.mark.skipif(
        not __import__("ringkernel").is_cuda_available(), reason="CUDA not available"
    )
    def test_gpu_pool_config(self):
        import ringkernel

        config = ringkernel.cuda.PyGpuPoolConfig.for_graph_analytics()
        assert config.track_allocations

    @pytest.mark.skipif(
        not __import__("ringkernel").is_cuda_available(), reason="CUDA not available"
    )
    def test_stream_config(self):
        import ringkernel

        config = ringkernel.cuda.PyStreamConfig.performance()
        assert config.num_compute_streams == 4

    @pytest.mark.skipif(
        not __import__("ringkernel").is_cuda_available(), reason="CUDA not available"
    )
    def test_enumerate_devices(self):
        import ringkernel

        devices = ringkernel.cuda.enumerate_devices()
        assert len(devices) >= 1
        assert devices[0].ordinal == 0


class TestBenchmark:
    """Tests for benchmark functionality."""

    def test_config_presets(self):
        import ringkernel

        quick = ringkernel.benchmark.PyBenchmarkConfig.quick()
        comprehensive = ringkernel.benchmark.PyBenchmarkConfig.comprehensive()

        assert quick.warmup_iterations < comprehensive.warmup_iterations
        assert quick.measurement_iterations < comprehensive.measurement_iterations

    def test_suite_creation(self):
        import ringkernel

        config = ringkernel.benchmark.PyBenchmarkConfig.quick()
        suite = ringkernel.benchmark.PyBenchmarkSuite(config)

        assert suite.is_empty()
        assert len(suite) == 0

    def test_result_creation(self):
        import ringkernel

        result = ringkernel.benchmark.PyBenchmarkResult(
            workload_id="test",
            size=1000,
            total_time_secs=0.5,
        )

        assert result.workload_id == "test"
        assert result.size == 1000
        assert result.throughput_ops == 2000.0  # 1000 / 0.5

    def test_suite_add_result(self):
        import ringkernel

        config = ringkernel.benchmark.PyBenchmarkConfig.quick()
        suite = ringkernel.benchmark.PyBenchmarkSuite(config)

        result = ringkernel.benchmark.PyBenchmarkResult(
            workload_id="test",
            size=1000,
            total_time_secs=0.5,
        )

        suite.add_result(result)
        assert len(suite) == 1

    def test_result_from_measurements(self):
        import ringkernel

        result = ringkernel.benchmark.PyBenchmarkResult.from_measurements(
            workload_id="measured",
            size=1000,
            measurement_times_secs=[0.1, 0.11, 0.09, 0.1, 0.1],
        )

        assert result.workload_id == "measured"
        assert result.size == 1000
        assert result.throughput_mops() > 0

    def test_baseline_creation(self):
        import ringkernel

        config = ringkernel.benchmark.PyBenchmarkConfig.quick()
        suite = ringkernel.benchmark.PyBenchmarkSuite(config)

        result = ringkernel.benchmark.PyBenchmarkResult(
            workload_id="test",
            size=1000,
            total_time_secs=0.5,
        )
        suite.add_result(result)

        baseline = suite.create_baseline("v1.0.0")
        assert baseline.version == "v1.0.0"
        assert len(baseline) == 1

    def test_report_generation(self):
        import ringkernel

        config = ringkernel.benchmark.PyBenchmarkConfig.quick()
        suite = ringkernel.benchmark.PyBenchmarkSuite(config)

        result = ringkernel.benchmark.PyBenchmarkResult(
            workload_id="test",
            size=1000,
            total_time_secs=0.5,
        )
        suite.add_result(result)

        md = suite.generate_markdown_report()
        assert "test" in md
        # Size may be comma-formatted as "1,000"
        assert "1,000" in md or "1000" in md
