#!/usr/bin/env python3
"""坐标转换单元测试：ENU <-> NED 必须互为逆运算。

运行：python3 test_transforms.py
"""
import sys
import importlib.util
import types


def load_class(path):
    spec = importlib.util.spec_from_file_location("uav_tracker_mod", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["uav_tracker_mod"] = mod
    spec.loader.exec_module(mod)
    return mod.UavTracker


def main():
    path = sys.argv[1]
    UavTracker = load_class(path)
    node = object.__new__(UavTracker)   # 不初始化 rclpy

    # ENU (东, 北, 上) -> NED (北, 东, 下)
    assert node.enu_to_ned(1.0, 2.0, 3.0) == (2.0, 1.0, -3.0), "ENU->NED 映射错误"
    assert node.enu_to_ned(0.0, 0.0, 0.0) == (0.0, 0.0, 0.0), "原点应映射到原点"

    # 北方向：ENU 的 +y 应变成 NED 的 +x
    n, e, d = node.enu_to_ned(0.0, 10.0, 0.0)
    assert (n, e) == (10.0, 0.0), "ENU 北(+y) 应映射到 NED 北(+x)"

    # 东方向：ENU 的 +x 应变成 NED 的 +y
    n, e, d = node.enu_to_ned(10.0, 0.0, 0.0)
    assert (n, e) == (0.0, 10.0), "ENU 东(+x) 应映射到 NED 东(+y)"

    # 高度：ENU 上为正，NED 下为正
    _, _, d = node.enu_to_ned(0.0, 0.0, 5.0)
    assert d == -5.0, "ENU 高度(+) 应映射到 NED 下(-)"

    # 往返一致性
    for (x, y, z) in [(1.5, -2.5, 3.5), (-10.0, 20.0, 0.0), (0.0, 0.0, -1.0)]:
        back = node.ned_to_enu(*node.enu_to_ned(x, y, z))
        assert back == (x, y, z), f"往返转换不一致: {(x, y, z)} -> {back}"

    print("ALL TESTS PASSED")


if __name__ == "__main__":
    main()
