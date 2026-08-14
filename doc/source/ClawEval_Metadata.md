# ClawEval 任务与数据集 Metadata

> 来源：[GitHub](https://github.com/claw-eval/claw-eval) | [HuggingFace](https://huggingface.co/datasets/claw-eval/Claw-Eval) | [论文 arXiv:2604.06132](https://arxiv.org/abs/2604.06132)

---

## 1. 总览

| 属性 | 值 |
|------|-----|
| 总任务数 | 300 |
| 评测维度 | Completion, Safety, Robustness |
| 评测指标 | Pass³（三次独立运行全部通过才算通过） |
| 评分公式 | $score = s_{safety} \times (0.8 \cdot s_{completion} + 0.2 \cdot s_{robustness})$ |
| Rubric 总数 | 2,159 |
| 语言 | English / Chinese |
| 许可证 | MIT |

---

## 2. 三大 Split 及 9 个 Category（论文 Table 2）

| Group | Category | 描述 | 数量 | 占比 |
|-------|----------|------|------|------|
| **General (161)** | Easy | 单服务查询、基础调度 | 71 | 23.7% |
| | Medium | 跨服务协调、数据检索 | 47 | 15.7% |
| | Hard | 多系统编排、金融合规、运维 | 43 | 14.3% |
| **Multimodal (101)** | Video | 简单 QA、视频定位 | 53 | 17.7% |
| | Doc & Image | 图表解读、跨页推理 | 22 | 7.3% |
| | Code | 网页生成、SVG 动画、视频编辑 | 26 | 8.7% |
| **Multi-turn (38)** | STEM | 数据分析、科学推理 | 10 | 3.3% |
| | Social Science | 法律、教育、公共政策 | 13 | 4.3% |
| | Business | 金融、投资、企业战略 | 15 | 5.0% |
| **Total** | | | **300** | **100%** |

---

## 3. 纯文本任务（General Split）详情

### 3.1 难度分布

| 难度 | 数量 | 占比 | 典型任务 |
|------|------|------|---------|
| Easy | 71 | 44.1% | 单服务查询、基础调度 |
| Medium | 47 | 29.2% | 跨服务协调、数据检索 |
| Hard | 43 | 26.7% | 多系统编排、金融合规、运维 |

### 3.2 领域分布（从 C-series 任务目录推断）

> **⚠️ 证据缺口（C4）**：本小节的领域分组是从 `tasks/` 目录的 **C-series** 任务名**推断**的。但 C-series 共 38 个任务，**属于 Multi-turn split，并非 General split**——因此下表**不能**直接作为 General split「24 个 category × 各自数量」的依据，仅作领域覆盖的**直觉参考**。General split 的 161 个任务的真实 category 字段必须从 HuggingFace 数据集加载后统计（见 §「未决数据缺口」）。在 manifest 落地前，buffer 的 9 桶映射（见 `CLAUDE.md` / `runs/_analysis/capability_buckets/buckets.json`）不应以本表为准。

论文提到 General split 覆盖 **24 个 category**，但未完整列出。以下从 `tasks/` 目录的 C-series（38 个任务，Multi-turn split）推断领域分组（**非 General split 证据**）：

| 领域 | 任务数 | 任务列表 |
|------|--------|---------|
| **Finance** | 8 | C01 mortgage_prepay, C02 personal_finance, C03 real_estate_finance, C05 personal_finance_2, C07 financial_valuation, C08 personal_finance_3, C35 income_tax_calculation, C36 insurance_planning, C37 income_tax_practice |
| **Legal / Compliance** | 3 | C10 labor_law, C14 cross_border_compliance, C24 fire_safety_code |
| **Medical / Health** | 4 | C06 clinical_pharmacy, C20 mental_health_social_work, C26 public_health, C29 pediatric_medication |
| **Data / Statistics** | 5 | C13 psychology_statistics, C18 statistical_analysis_spss, C19 ab_testing_sample_size, C21 ecommerce_data_analysis, C27 data_automation, C30 data_troubleshooting |
| **Engineering / DevOps** | 3 | C11 container_orchestration, C17 devops_sop_design, C28 distributed_systems |
| **Operations / Business** | 5 | C12 ecommerce_operations, C16 hr_workforce_planning, C31 restaurant_management, C32 saas_metrics, C38 inventory_management |
| **Writing / Design** | 3 | C23 academic_writing, C33 accessibility_design, C34 product_design |
| **Psychology** | 2 | C22 experimental_psychology, C13 psychology_statistics |
| **Construction** | 2 | C15 structural_seismic_design, C25 construction_cost |
| **Media** | 2 | C04 image_processing, C09 ai_video_creation |

> ⚠️ 注意：C-series 共 38 个任务，属于 multi-turn split。General split 的 161 个任务的完整 category 列表需要从 HuggingFace 数据集加载后获取。

### 3.3 语言分布

C-series 中：
- **中文 (zh)**：28 个（73.7%）
- **英文 (en)**：10 个（26.3%）

---

## 4. Multimodal Split 详情（101 个任务）

| 子领域 | 任务数 | 任务范围 |
|--------|--------|---------|
| **Video QA** | 53 | 电影识别、体育 QA（羽毛球/网球/乒乓球/斯诺克）、监控、食物、健身、购物 |
| **Doc & Image** | 22 | 折线图/雷达图/柱状图提取、参考文献验证 |
| **Code (生成)** | 26 | 网页重建、SVG 动画（时钟/太阳系/烟花）、乐谱生成、地铁图 |

### Multimodal 各子领域 Pass³ 对比（论文 Table 5）

| 模型 | Video (53) | Doc & Image (22) | Code (26) | Overall |
|------|-----------|-----------------|-----------|---------|
| GPT 5.4 | 11.5 | **54.5** | 29.6 | **25.7** |
| Claude Opus 4.6 | **15.4** | 45.5 | 25.9 | 24.8 |
| Claude Sonnet 4.6 | **15.4** | 40.9 | 25.9 | 23.8 |

> Video 任务最难（平均 Pass³ 仅 10.7%），Doc & Image 相对最易（32.3%）。

---

## 5. Multi-turn Split 详情（38 个任务）

| Category | 数量 | 描述 |
|----------|------|------|
| STEM | 10 | 数据分析、科学推理 |
| Social Science | 13 | 法律、教育、公共政策 |
| Business | 15 | 金融、投资、企业战略 |

- 使用模拟用户画像（simulated user persona）进行多轮对话
- 评分由 Claude Opus 4.6 担任 judge（General/Multimodal 用 Gemini 3 Flash）
- 关键发现：提问精度解释 76% 的 Pass³ 方差（r=0.87），轮数与表现几乎无关（r=0.07）

---

## 6. 完整任务目录（100 个任务）

### C-Series：专业领域任务（38 个，属于 Multi-turn split）

| 编号 | 任务 ID | 领域 | 语言 |
|------|---------|------|------|
| C01 | mortgage_prepay | Finance | zh |
| C02 | personal_finance | Finance | zh |
| C03 | real_estate_finance | Finance | en |
| C04 | image_processing | Media | zh |
| C05 | personal_finance_2 | Finance | zh |
| C06 | clinical_pharmacy | Medical | zh |
| C07 | financial_valuation | Finance | zh |
| C08 | personal_finance_3 | Finance | zh |
| C09 | ai_video_creation | Media | zh |
| C10 | labor_law | Legal | zh |
| C11 | container_orchestration | Engineering | zh |
| C12 | ecommerce_operations | Operations | zh |
| C13 | psychology_statistics | Data/Stats | zh |
| C14 | cross_border_compliance | Legal | zh |
| C15 | structural_seismic_design | Construction | en |
| C16 | hr_workforce_planning | Operations | en |
| C17 | devops_sop_design | Engineering | en |
| C18 | statistical_analysis_spss | Data/Stats | en |
| C19 | ab_testing_sample_size | Data/Stats | en |
| C20 | mental_health_social_work | Medical | zh |
| C21 | ecommerce_data_analysis | Data/Stats | zh |
| C22 | experimental_psychology | Psychology | zh |
| C23 | academic_writing | Writing | zh |
| C24 | fire_safety_code | Legal | zh |
| C25 | construction_cost | Construction | zh |
| C26 | public_health | Medical | zh |
| C27 | data_automation | Data/Stats | zh |
| C28 | distributed_systems | Engineering | zh |
| C29 | pediatric_medication | Medical | zh |
| C30 | data_troubleshooting | Data/Stats | zh |
| C31 | restaurant_management | Operations | zh |
| C32 | saas_metrics | Operations | zh |
| C33 | accessibility_design | Design | zh |
| C34 | product_design | Design | zh |
| C35 | income_tax_calculation | Finance | zh |
| C36 | insurance_planning | Finance | zh |
| C37 | income_tax_practice | Finance | zh |
| C38 | inventory_management | Operations | zh |

### M-Series：多模态任务（62 个，属于 Multimodal split）

| 编号 | 任务 ID | 子领域 |
|------|---------|--------|
| M001 | clock | SVG 动画 |
| M002 | world_clock | SVG 动画 |
| M003 | solar_system | SVG 动画 |
| M004 | countdown_fireworks | SVG 动画 |
| M005 | score_canon | 乐谱生成 |
| M006 | score_mariage | 乐谱生成 |
| M007 | score_symphony | 乐谱生成 |
| M008 | metro_map_1 | 地铁图 |
| M009 | metro_map_2 | 地铁图 |
| M010 | score_canon_animated | 动态乐谱 |
| M011 | score_mariage_animated | 动态乐谱 |
| M012 | score_symphony_animated | 动态乐谱 |
| M013 | metro_route_1 | 地铁路线 |
| M014 | metro_route_2 | 地铁路线 |
| M015 | video_subtitle_ocr_english | 视频 OCR |
| M016 | video_subtitle_ocr_chinese_filter | 视频 OCR |
| M017 | video_subtitle_ocr_timestamp | 视频 OCR |
| M018 | doc_extraction_line_chart | 文档提取 |
| M019 | doc_extraction_radar_chart | 文档提取 |
| M020 | multi_doc_extraction_bar_chart | 文档提取 |
| M021 | doc_reference_verification | 文档提取 |
| M022 | video_movie_recognition | 视频理解 |
| M023 | video_paper_understanding | 视频理解 |
| M024 | video_factory_promo_webpage | 视频→网页 |
| M025 | video_badminton_match_qa | 体育 QA |
| M026 | video_story_interactive_webpage | 视频→网页 |
| M027 | video_food_memo | 视频理解 |
| M028 | video_badminton_score_chart | 体育分析 |
| M029 | video_surveillance_clip | 监控 |
| M030 | video_snack_checklist | 视频理解 |
| M031 | video_room_floorplan | 视频→平面图 |
| M032 | video_tennis_rally_qa | 体育 QA |
| M033 | video_tennis_breakpoint_qa | 体育 QA |
| M034 | video_tennis_shotlog_qa | 体育 QA |
| M035 | video_tennis_exhibition_qa | 体育 QA |
| M036 | video_butterfly_drawing_tutorial | 视频理解 |
| M037 | video_food_shop_search | 视频理解 |
| M038 | video_lvb_hill_descent | 视频理解 |
| M039 | video_lvb_machine_dog | 视频理解 |
| M040 | video_lvb_vehicle_identification | 视频理解 |
| M041 | video_lvb_artwork_scene | 视频理解 |
| M042 | video_mme_multihop_reasoning | 视频推理 |
| M043 | video_mme_device_identification | 视频识别 |
| M044 | video_mme_bugatti_identification | 视频识别 |
| M045 | video_mme_building_identification | 视频识别 |
| M046 | video_mme_news_segments | 视频理解 |
| M047 | video_fitness_exercise_summary | 体育分析 |
| M048 | video_fitness_pullup_frames | 体育分析 |
| M049 | video_phone_comparison | 视频理解 |
| M050 | video_shopping_receipt | 视频理解 |
| M051 | video_surveillance_intrusion | 监控 |
| M052 | webpage_recreation | 网页重建 |
| M053 | video_badminton_rally_count | 体育分析 |
| M054 | video_badminton_match_analysis | 体育分析 |
| M055 | video_badminton_baseline_out | 体育分析 |
| M056 | video_badminton_net_error | 体育分析 |
| M057 | video_pingpong_rally_count | 体育分析 |
| M058 | video_pingpong_serve_stats | 体育分析 |
| M059 | video_pingpong_smash_ace | 体育分析 |
| M060 | video_pingpong_let_serve | 体育分析 |
| M061 | video_snooker_clearance_sequence | 体育分析 |
| M062 | video_snooker_brown_ball_time | 体育分析 |

---

## 7. 评测结果参考（论文 Table 3）

### General + Multi-turn 综合排名

| 排名 | 模型 | General Score | Multi-turn Score | Overall Score | Overall Pass³ |
|------|------|--------------|-----------------|---------------|---------------|
| 1 | Claude Sonnet 4.6 | **81.3** | **81.9** | **81.4** | 67.8 |
| 2 | Claude Opus 4.6 | 80.6 | 79.6 | 80.4 | **70.4** |
| 3 | GPT 5.4 | 78.3 | 79.0 | 78.4 | 60.3 |
| 4 | Gemini 3.1 Pro | 76.6 | 80.2 | 77.3 | 57.8 |
| 5 | MiMo V2 Pro | 76.0 | 81.0 | 77.0 | 57.8 |

> 最强模型 Overall Pass³ 仅 70.4%，说明任务整体难度较高。

---

## 8. 工具能力层（论文 Table 6）

| 功能组 | 工具 | 用途 |
|--------|------|------|
| **系统层** | | |
| 代码执行 | Bash | 执行 shell 命令 |
| 文件操作 | Read, Write, Edit | 读取、创建、修改文件 |
| 代码搜索 | Glob, Grep | 查找文件、正则搜索内容 |
| Web 交互 | BrowserScreenshot, WebSearch, WebFetch | 截图、搜索、抓取网页 |
| 多模态媒体 | ReadMedia, Download | 处理视频/图片/PDF、下载 |
| **服务层** | | |
| 任务 API | 每个任务自定义工具 | 与模拟服务交互 |

---

## 9. 待补充信息

- [ ] General split 的 24 个 category 完整列表及各自数量（需从 HuggingFace 数据集加载后统计 `category` 字段）
- [ ] General split 各 category 的语言分布
- [ ] 各 category 的平均 reward / Pass³ 分布
