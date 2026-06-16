"""
TRM forward() 走查演示
========================
用一个具体例子展示 TRM 完整推理流程。
不实现真实运算，用 print 和 mock 数据展示每一步的输入输出。

场景: "杯子从桌上掉下来碎了，但是桌子明明很矮，不应该碎才对。"
"""

from dataclasses import dataclass, field
from typing import List, Optional
from enum import Enum


# ----- 迷你数据定义（复用 TRM_theory.py 的结构）-----

class ChunkDecision(Enum):
    CONTINUE = "continue"
    STOP = "stop"

class Intent(Enum):
    UPDATE_TEMPLATES = "update_templates"
    EXECUTE_ACTION = "execute_action"
    OUTPUT_LANGUAGE = "output_language"

@dataclass
class AccumulatedState:
    closed_conclusions: List[str] = field(default_factory=list)
    open_questions: List[str] = field(default_factory=list)
    chain_depth: int = 0
    triggered_templates: List[str] = field(default_factory=list)
    dangling_span: Optional[str] = None

@dataclass
class TemplateSkeleton:
    template_id: str
    trigger: List[str]       # 简化为谓词列表
    slots: List[str]
    expansion: str
    is_atomic: bool
    original_text: str

@dataclass
class ChunkOutput:
    incremental_chain: List[str]
    updated_state: AccumulatedState
    chunk_decision: ChunkDecision
    intents: List[Intent]

@dataclass
class ReasoningOutput:
    full_chain: List[str]
    final_state: AccumulatedState
    intents: List[Intent]
    structured_conclusion: str


# ----- Mock 模板库 -----

TEMPLATES = {
    "fall_damage": TemplateSkeleton(
        template_id="fall_damage",
        trigger=["lose_support(?X, ?Y)", "fall(?X)", "fragile(?X)"],
        slots=["?X: physical_object, brittle", "?Y: surface, elevated"],
        expansion="""
步骤1: 确认 ?X 是否 lose_support(?X, ?Y)
步骤2: 确认 ?X 是否发生 fall
步骤3: 检查 ?Y 的表面属性（硬/软）
步骤4: 若 fall + hard_surface(?Y) → 推断 ?X 受损（闭合）
       若 fall + soft_surface(?Y) → 展开子模板 "soft_landing"
       若 ?Y 高度异常低 → 展开子模板 "height_anomaly"
""",
        is_atomic=False,
        original_text="如果物体失去支撑从高处坠落碰撞硬表面，则物体会损坏。",
    ),
    "height_anomaly": TemplateSkeleton(
        template_id="height_anomaly",
        trigger=["fall(?X)", "low_height(?Y)", "damaged(?X)", "contradiction(expected_safe, actual_damage)"],
        slots=["?X: physical_object", "?Y: surface"],
        expansion="""
步骤1: 确认矛盾——预期安全但实际受损
步骤2: 检查 ?X 本身是否存在先存缺陷（pre-existing flaw）
步骤3: 检查 ?X 着地姿态（角/边/面）
步骤4: 若先存缺陷 → 归因于缺陷，闭合
       若特殊着地姿态 → 应力集中导致 → 归因于姿态，闭合
       若都排除 → 原因未知，标记 open
""",
        is_atomic=False,
        original_text="如果物体从异常低的高度坠落却损坏，需要检查物体先存缺陷或特殊着地姿态。",
    ),
    "pre_existing_flaw": TemplateSkeleton(
        template_id="pre_existing_flaw",
        trigger=["has_flaw(?X)", "damaged(?X)", "triggered_by_minor_impact(?X)"],
        slots=["?X: physical_object"],
        expansion="步骤1: 确认 ?X 有先存微观缺陷 → 闭合（原子模板）",
        is_atomic=True,
        original_text="如果物体存在先存缺陷，轻微冲击即可触发损坏。",
    ),
}


# ----- 模拟编码端输出 -----

def mock_encode(chunk_text: str):
    """模拟 Transformer 编码端输出。
    实际返回的是 [n × d_model] 矩阵，
    这里用提取出的关键词和关系来模拟。"""
    
    # 模拟: 编码端看了 chunk_text，提取出了这些关系骨架
    print(f"    [编码端] 输入文本: \"{chunk_text}\"")
    print(f"    [编码端] 输出: [n × d_model] 矩阵（此处省略数值）")
    
    # 模拟结构抽取结果
    entities = []
    relations = []
    
    if "杯子" in chunk_text and "桌上" in chunk_text:
        entities = ["杯子(physical_object)", "桌子(surface)"]
        relations = ["lose_support(杯子, 桌子)", "fall(杯子)", "fragile(杯子)"]
    if "碎了" in chunk_text:
        relations.append("damaged(杯子)")
    if "矮" in chunk_text:
        relations.append("low_height(桌子)")
    if "不应该" in chunk_text or "不应该碎" in chunk_text:
        relations.append("contradiction(expected_safe, actual_damage)")
    
    print(f"    [结构抽取] 实体: {entities}")
    print(f"    [结构抽取] 关系三元组: {relations}")
    
    return entities, relations


# ----- 模板匹配模拟 -----

def mock_template_search(relations: List[str], state: AccumulatedState):
    """模拟模板检索匹配。
    实际是 StructuredQuery vs 模板 StructuredQuery 的 Cross-Attention，
    这里用规则模拟匹配结果。"""
    
    matched = []
    
    for tid, tmpl in TEMPLATES.items():
        # 简化匹配: 检查所有 trigger 谓词是否都存在
        hits = sum(1 for t in tmpl.trigger if any(t.split("(")[0] in r for r in relations))
        total = len(tmpl.trigger)
        score = hits / total if total > 0 else 0
        
        if score > 0.4:  # 阈值
            matched.append((tmpl, score))
    
    matched.sort(key=lambda x: x[1], reverse=True)
    
    print(f"\n    [模板检索] 输入关系数: {len(relations)}")
    for tmpl, score in matched:
        print(f"    [模板检索] 命中: {tmpl.template_id} (score={score:.2f})")
        print(f"              原文: \"{tmpl.original_text}\"")
    
    return matched


# ----- 停止控制器 -----

def mock_stop_decide(state: AccumulatedState, match_score: float, is_last: bool):
    """模拟停止判定"""
    reasons = []
    
    # 条件1: 问题全闭合
    if not state.open_questions:
        reasons.append("所有问题已闭合")
    
    # 条件2: 匹配分太低
    if match_score < 0.5:
        reasons.append(f"匹配分过低({match_score:.2f})")
    
    # 条件3: 深度上限
    if state.chain_depth >= 5:
        reasons.append("达到深度上限")
    
    # 边界: 最后一块
    if is_last:
        reasons.append("已是最后一块，强制停止")
        return True, [Intent.OUTPUT_LANGUAGE], reasons
    
    should_stop = len(reasons) > 0
    intents = [Intent.OUTPUT_LANGUAGE] if should_stop else []
    return should_stop, intents, reasons


# ═══════════════════════════════════════════════════════════════
# _trm_forward() 主函数
# ═══════════════════════════════════════════════════════════════

def _trm_forward() -> ReasoningOutput:
    """
    TRM 完整推理流程（走查版）
    
    输入:
      raw_text    用户输入的自然语言
      chunk_size  分块大小（字符数，演示用小值）
    
    输出:
      ReasoningOutput
    """
    raw_text = "杯子从桌上掉下来碎了，但是桌子明明很矮，不应该碎才对。"
    chunk_size: int = 15

    print("=" * 70)
    print(f"TRM forward() 开始")
    print(f"输入: \"{raw_text}\"")
    print("=" * 70)
    
    # ── 步骤 0: Tokenize + 分块 ──
    # 真实流程: raw_text → Tokenizer.encode → [N] token IDs → Chunker.split
    # 这里简化为按字符分块
    print("\n── 步骤 0: 分块 ──")
    chunks = []
    text = raw_text
    while text:
        chunk = text[:chunk_size]
        chunks.append(chunk)
        text = text[chunk_size:]
    
    # 悬空检测: 最后一个 chunk 末尾是否完整句子
    dangling = None
    if chunks and not chunks[-1].endswith(("。", "！", "？")):
        # 找最后一个完整句子的位置
        last_period = max(chunks[-1].rfind("。"), chunks[-1].rfind("！"), chunks[-1].rfind("？"))
        if last_period > 0:
            dangling = chunks[-1][last_period + 1:]
            chunks[-1] = chunks[-1][:last_period + 1]
    
    print(f"分块结果: {len(chunks)} 个 chunk, chunk_size={chunk_size} 字符")
    for i, c in enumerate(chunks):
        print(f"  Chunk_{i+1}: \"{c}\"")
    if dangling:
        print(f"  悬空片段: \"{dangling}\" → 带入 Chunk_2")
    
    # ── 初始化累积状态 ──
    state = AccumulatedState()
    full_chain = []
    
    # ── 逐块循环 ──
    for k, chunk_text in enumerate(chunks):
        kk = k + 1
        print(f"\n{'─' * 50}")
        print(f"── Chunk_{kk} / {len(chunks)} ──")
        print(f"   累积状态: 已闭合={len(state.closed_conclusions)}条, "
              f"未闭合={len(state.open_questions)}个, "
              f"深度={state.chain_depth}")
        
        # 悬空片段 prepend
        if state.dangling_span:
            chunk_text = state.dangling_span + chunk_text
            print(f"   悬空 prepend 后: \"{chunk_text}\"")
            state.dangling_span = None
        
        # ── 步骤 1: 编码 ──
        entities, relations = mock_encode(chunk_text)
        
        # ── 步骤 2: 模板检索 ──
        candidates = mock_template_search(relations, state)
        best_score = candidates[0][1] if candidates else 0.0
        
        # ── 步骤 3: 模板展开 ──
        print(f"\n    [展开引擎] 开始展开——")
        new_nodes = []
        
        for tmpl, score in candidates:
            print(f"    [展开引擎] 展开模板: {tmpl.template_id}")
            
            # 虚拟展开步骤
            steps = [s.strip() for s in tmpl.expansion.strip().split("\n") if s.strip().startswith("步骤")]
            for step in steps:
                step_text = step.split(":")[0] + f" ({tmpl.template_id})"
                new_nodes.append(step_text)
                print(f"               {step_text}")
            
            # 检查是否原子模板
            if tmpl.is_atomic:
                print(f"    [展开引擎] ⚡ 命中原子模板: {tmpl.template_id}")
                conclusion = f"结论[{tmpl.template_id}]: 杯子存在先存微观缺陷，轻微冲击即可碎裂。"
                new_nodes.append(conclusion)
                state.closed_conclusions.append(conclusion)
                state.open_questions = []  # 问题已闭合
            
            # 检查 height_anomaly 模板的展开
            if tmpl.template_id == "height_anomaly" and not state.closed_conclusions:
                state.open_questions.append("杯子是否先存缺陷？")
                state.open_questions.append("着地姿态是否为角/边接触？")
                print(f"    [展开引擎] ❓ 提出待验证问题: {state.open_questions}")
            
            state.triggered_templates.append(tmpl.template_id)
            state.chain_depth += 1
        
        full_chain.extend(new_nodes)
        
        # ── 步骤 4: 停止判定 ──
        is_last = (kk == len(chunks))
        should_stop, intents, stop_reasons = mock_stop_decide(
            state, best_score, is_last
        )
        
        print(f"\n    [停止判定] 是否停止: {should_stop}")
        for r in stop_reasons:
            print(f"              原因: {r}")
        
        if should_stop:
            # ── 意图分流 ──
            print(f"\n── 意图分流 ──")
            print(f"   意图: {[i.value for i in intents]}")
            
            structured_conclusion = ""
            if Intent.OUTPUT_LANGUAGE in intents:
                if state.closed_conclusions:
                    structured_conclusion = (
                        "杯子碎裂的原因是：虽然桌面较低，但杯子本身存在先存微观缺陷，"
                        "在低高度坠落时缺陷处应力集中导致碎裂。"
                        "正常高度的坠落不足以损坏完好的杯子，因此「矮桌碎了」构成异常信号，"
                        "指向杯子本身存在缺陷。"
                    )
                else:
                    structured_conclusion = "无法确定杯子碎裂的原因，需要更多信息（杯子材质、着地姿态）。"
                
                print(f"\n── 解码端生成 ──")
                print(f"   结构化结论: \"{structured_conclusion}\"")
                print(f"   解码端 → 自然语言输出 ↓")
                print(f"\n   🗣️  {structured_conclusion}")
            
            # ── 汇总输出 ──
            print(f"\n{'=' * 70}")
            print("TRM forward() 完成")
            print(f"推理链: {len(full_chain)} 步")
            print(f"触发模板: {state.triggered_templates}")
            print(f"已闭合结论: {state.closed_conclusions}")
            print(f"深度: {state.chain_depth}")
            print("=" * 70)
            
            return ReasoningOutput(
                full_chain=full_chain,
                final_state=state,
                intents=intents,
                structured_conclusion=structured_conclusion,
            )
        
        # CONTINUE: 更新状态，准备下一个 chunk
        # 悬空片段: 检查当前 chunk 的结尾
        if not chunk_text.endswith(("。", "！", "？")) and not is_last:
            last_period = max(chunk_text.rfind("。"), chunk_text.rfind("！"), chunk_text.rfind("？"))
            if last_period > 0:
                state.dangling_span = chunk_text[last_period + 1:]
    
    # 循环结束但未 STOP（理论不应到达，边界保护）
    return ReasoningOutput(
        full_chain=full_chain,
        final_state=state,
        intents=[Intent.OUTPUT_LANGUAGE],
        structured_conclusion="推理未完成（闭合不完整）。",
    )


# ═══════════════════════════════════════════════════════════════
# 运行演示
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    result = _trm_forward()
