<!--
感谢贡献！请只勾选本 PR 已实际完成的项目；不适用或跳过的项目请写明原因和跟踪 Issue。
可在文末“其他补充”区域自由增加项目、方案、测试结果或风险说明。
-->

## 变更摘要

<!-- 做了什么、为什么做；关联 Issue：Closes # / Related # -->

## 变更类型

- [ ] `[Bugfix]` Bug 修复
- [ ] `[Feature]` 新功能/新算子
- [ ] `[Refactor]` 重构
- [ ] `[CI]` CI 或构建
- [ ] `[Skills]` Skill/开发流程
- [ ] `[Testing]` 测试
- [ ] `[Example]` 示例/Benchmark
- [ ] 其他：<!-- 填写类型 -->

## 1. 算子合入标准

- [ ] 新增算子在默认场景下通过：未显式指定 `target="ascendc"`，PTO 后端已完成编译、运行和精度验证；如不适用，已说明原因。
- [ ] 适用时 AscendC 与 PTO 双后端均已通过；不支持的 dtype/shape/后端行为已显式说明，没有静默跳过。
- [ ] 未影响现有算子用例、公共接口和已有后端行为。
- [ ] `examples/` 下的算子有性能要求和代表性 shape；历史遗留且暂未达标的实现已迁移到 `examples_experiment/`，或有维护者批准的豁免。
- [ ] 有 AscendC 参考实现时，TileLang 性能达到 AscendC 的 50% 以上；如有 GPU 基线，候选耗时不超过其 2 倍。
- [ ] 无 AscendC 参考实现时，已检查 [性能反模式清单](../.agents/skills/tilelang-perf-optimization/references/performance-antipatterns.md)，并记录保留项及原因。
- [ ] 优先使用 Developer 原语；使用 Expert/Mixed、手写 intrinsic 或特殊实现时，已说明原因和验证结果。
- [ ] PR 已附精度/性能测试报告，以及典型 shape 的 simulator 结果截图；报告包含环境、shape、baseline、候选结果、单位和复现命令。
- [ ] 新增单算子的全量测试用例耗时不超过 2 分钟；若超过，已通过 pytest marker 仅保留核心用例在 PR 阶段执行，其余用例纳入每日/定时全量，并记录耗时和标签。

性能摘要/报告链接：<!-- 填写结果或附件；不适用时说明原因 -->

### Pytest 接入修改细则

新增或迁移算子测试时，按以下步骤修改。可参考已合入的 [batch_gemm 测试](https://github.com/tile-ai/tilelang-ascend/blob/ascendc_pto/examples/batch_gemm/test_batch_gemm.py)。

1. **补齐测试入口。** 在算子目录的 `test_*.py` 中提供 pytest 可收集的 `test_*` 函数，复用已有 `run_*` 或验证函数。仅改文件名不能完成迁移；带 NPU 初始化、参数解析或执行副作用的源模块，应在测试函数或 fixture 内加载，避免收集阶段运行算子。
2. **让失败传递给 pytest。** 对明确返回布尔状态的函数使用断言；返回元组时，按返回值定义取出通过状态后断言。内部已做数值断言、成功返回 `None` 的函数直接调用，不要机械地写成 `assert run_*()`。直接比较输出时使用 `torch.testing.assert_close` 等数值断言。不能只打印结果、返回 `False` 或捕获异常后继续判通过。在 pytest 进程内复用测试逻辑，避免用子进程包装后丢失现有覆盖率采集。
3. **接好初始化并隔离副作用。** 将必要的设备设置和随机种子接入测试调用路径；pytest 不会自动执行 `__main__`。需要显式使用 `runpy` 且捕获 `SystemExit` 时，仅接受退出码为 `None` 或 `0`，其他退出码和断言失败必须继续抛出。导入时有 `parse_args()` 的模块，应暂存并替换 `sys.argv`，在 `finally` 中恢复；对临时修改的默认设备、导入路径等状态也要恢复。源模块会清理编译缓存时，应先在 fixture 中隔离缓存目录，结束后恢复，避免并行用例互相删除编译产物；可参考 [缓存隔离样例](https://github.com/tile-ai/tilelang-ascend/blob/ascendc_pto/examples/tile_kernels/quant/test_per_block_cast_lossless_kernel.py)。
4. **明确实际执行范围。** 按现有 marker 规则在 PR 阶段保留 L0 和关键回归，非核心扩展组合标记 `low_priority`；benchmark 和原本非阻塞的 Boundary 检查应明确安排。脚本的 `--level l0` 默认值不会限制 pytest 收集，不要将所有 `run_*` 批量改名后直接接入 CI。迁移前后应逐项核对用例、参考实现和容差；有意改变检查范围或精度标准时，说明差异及理由。
5. **配套核对 manifest。** 作者列出源文件与测试文件的仓库相对路径，由作者或维护者核对 `ci/operator_test_manifest.yaml` 中的映射；已有预留条目直接沿用，路径变更时同步更新。源文件与测试文件须为不同文件，且两者均存在时映射才生效。登记生效后，legacy runner 的 Python 脚本收集会跳过两者，测试交给 pytest；还应检查外层 `.sh` 或自定义任务是否再次手动调用它们，避免重复执行或遗漏原检查。
6. **附实际验证结果。** 在目标 Ascend NPU 环境先确认至少选中一个用例，再按对应 CI 通道的路径、marker 和并发数运行。核对 manifest 校验通过、目标测试出现在 `list-tests` 中，并附环境、完整命令、选中及通过/失败/跳过数量和耗时。收集成功不能作为精度通过；扩展用例另记录全量运行结果。

例如，`run_l0_tests()` 已完成必要初始化并明确返回布尔状态时：

```python
def test_operator_l0():
    assert run_l0_tests(), "L0 correctness check failed"
```

在仓库根目录，将下列路径替换为实际测试文件路径：

```bash
TEST_FILE="examples/your_operator/test_your_operator.py"
python -m pytest "$TEST_FILE" --collect-only -q -m "not (low_priority or ci_skip)"
python scripts/ci/resolve_operator_tests.py validate
python scripts/ci/resolve_operator_tests.py list-tests
# 当前 CI 的迁移算子批次使用 8 个 worker，增量 pytest 通道使用 4 个；
# 按本次对应通道设置，具体以 .github/workflows/ci_cd.yml 为准。
PYTEST_WORKERS=8
python -m pytest "$TEST_FILE" --forked -v -n "$PYTEST_WORKERS" -m "not (low_priority or ci_skip)"
# 单独验证每日/全量范围；保留 low_priority，仅排除 ci_skip。
python -m pytest "$TEST_FILE" --forked -v -n "$PYTEST_WORKERS" -m "not ci_skip"
```

覆盖率路径由 `examples/bench_test.sh` 逐文件执行，配置不同于上述并发精度测试；需要覆盖率结果时，另按该路径验证。

## 2. 框架及接口代码合入标准

- [ ] 框架或接口修改已在双后端/默认 target 下验证，不影响现有算子用例；能力边界已明确。

> **测试门禁（重点）｜PR 快速、每日完整：** 只将最小且可复现的核心 ST/UT 放入 PR 阶段；非核心扩展组合通过 marker 放入每日/定时全量任务，避免无意义泛化浪费 CI 资源。

- [ ] 已新增或更新**代表性 ST/UT**，覆盖默认 target、关键边界和适用后端。
- [ ] 算子及对应测试用例已在本地多轮重复验证，未发现概率性失败；如出现偶发问题，已定位修复或在“其他补充”中说明。
- [ ] PR 阶段只保留核心冒烟/L0 与关键回归用例；核心回归不得用 marker 隐藏。
- [ ] 非核心扩展组合统一标记 `low_priority`，由每日/全量任务使用 `-m "not ci_skip"` 执行；`ci_skip` 仅用于已有 Issue 的已知不稳定或硬件限制。
- [ ] 已避免无意义泛化、重复编译和重复执行；涉及算子 pytest 接入时已完成第 1 节的修改和验证，不适用时已说明原因。
- [ ] 已更新对应文档，明确 API、后端、dtype、shape 或硬件能力边界；无文档变更时已说明原因。

## 3. PR 提交规范

- [ ] Commit message 使用英文，并包含类型标签：`[Bugfix]`、`[Feature]`、`[Refactor]`、`[CI]`、`[Skills]`、`[Testing]` 或 `[Example]`。
- [ ] 已检查 `git diff`/`git status`，PR 仅包含必要修改；CI 和 reviewer 意见已处理。

> **提交流程注意事项：** 代码检查和用例批跑流水线通过后，再发给 committer 检视/合入。
>
> **检视意见注意事项：** 对检视意见逐条回复；需要修改时完成修改，并重新运行相关检查和用例。

## 其他补充（可灵活增加）

<!-- 可复制以下内容，按需补充测试/检查命令与结果、方案设计、额外自检、风险、回退方式或已知限制。 -->

- [ ] 自定义检查：<!-- 内容、结果、证据/链接 -->

### 方案/风险/测试补充（可选）

<!-- 自由填写，不需要时删除本节。 -->

## 提交前确认

- [ ] 所有适用复选项均已完成；未完成、跳过或豁免项均已说明原因和跟踪项。
