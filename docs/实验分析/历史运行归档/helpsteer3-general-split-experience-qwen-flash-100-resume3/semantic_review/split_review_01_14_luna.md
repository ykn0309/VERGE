# 计划49拆分语义审查：第1–14次（Luna）

审查对象：`split_evidence_01_14.json` 中的实际归纳输入、完整案例 task/chosen/rejected、相关标准及 Judge 理由，以及原始接受输出和案例分组。判断不以局部 P>N 或两个输出文本是否矛盾为依据；允许单例组，考察是否存在可单独复用的设计/判分机制。

结论：**reasonable 4，unnecessary 10，uncertain 0**。
其中保留合理的分化：#1 背景事实核对与叙事结构、#3 信息局限校准与数学定义核验、#4 故事完成度与矛盾动机分析、#14 事件覆盖与角色—动作—对象绑定。其余拆分主要把同一项任务覆盖/事实落地原则按场景或粒度重复切开。

## 逐次结论

### #1 — reasonable — `exp_5002bcedf7f24c7280500a2dc6d24acb`

子经验：`exp_a0543332c22e4e928ad388678f773899`、`exp_71dc77f6c81544d5a21d9d86af6b036f`。

两组的评分目标不同：c7核对指定背景事实，2869考察叙事弧线与主角状态，作为独立规则拆分合理。但这不等于子经验文本无误：c7 原 task 明确要求故事“would start with the spaceship crashing”，因此“不要将未在首句集中陈述作为否决条件”必须受显式开篇约束限制，不能覆盖任务对位置/顺序的明确要求。

- `helpsteer3:train:c7c8461e3cb5da3a7150bf31ad2a4383107628b99fd87e182dc766a99be7c325`：Task 指定“3 months since...crashed”及“only survivor”；评审标准核对这些背景事实是否在故事中传达。
- `helpsteer3:train:c7c8461e3cb5da3a7150bf31ad2a4383107628b99fd87e182dc766a99be7c325`：原 task 开头要求“there would be dialogue in this story, which would start with the spaceship crashing”；chosen 却从三个月后的临产场景开始。
- `helpsteer3:train:2869b9bd2ddd218fd2da467d2ff5abbfb210a96bcfe328d184572113513abc80`：Task 是“a man finds himself running from a herd”；所引标准检查起因、发展、转折、结局和主角状态。

可兼容统一经验：可兼容表述：“故事评分分别核对prompt指定的背景事实（若未指定位置，可在全文任意位置清晰呈现）与核心叙事弧线；若prompt明确要求某事实/场景开篇或按特定顺序出现，则按该位置约束评分。”保留两条仍更利于分别检索。

案例/Judge/漂移：c7 的复合标准 Judge 布尔值 chosen=true、rejected=true，且 chosen_reason 自身反复修正“bruise”是否出现及判定；双胞胎/胎内通感标准的理由也与明确出现的文本不一致。原 task 还明确要求以飞船坠毁开场，chosen 从事后临产开始，而被引用的 Judge 标准未检查这一约束。因此，正信用不证明“不要将未在首句集中陈述作为否决条件”这一泛化文本正确；需保留“除非prompt明确要求开篇位置”的边界。

### #2 — unnecessary — `exp_a0543332c22e4e928ad388678f773899`

子经验：`exp_8083e675bd3f46f2924e8fe47dc5f582`、`exp_ccd386af56974af9813c301303e0af1a`。

两组都在回答故事是否呈现了prompt指定背景：旧案强调背景事实可通过不同段落/叙述方式清晰传达，新案要求对话不能替代缺失的地点等场景信息。这是“什么算清晰传达”的同一标准细化，可由一个经验明确区分背景事实、叙事位置和仅有对话的情况。

- `helpsteer3:train:c7c8461e3cb5da3a7150bf31ad2a4383107628b99fd87e182dc766a99be7c325`：Chosen 后段回忆坠毁并明确说 Duna 是唯一幸存者；任务的三个月时间信息仍未出现。
- `helpsteer3:train:e3d620cdd3af83d27be22276ca3cc1a5a2f5e74d7e5e51deda8d86ea2d1d503e`：Task 标注“In the school literature clubroom”；rejected 仅保留泛化的诗歌赞扬对话，没有地点或人物状态描写。

可兼容统一经验：核验prompt指定的背景事实是否在回答中清楚呈现，可由叙述、回忆或对话传达；若回答只有泛化对话而未传达地点/人物状态，仍算缺失。

案例/Judge/漂移：c7 的Judge原因自相矛盾（见拆分1）。e3 chosen 确实包含“literature club”“go around the room”及角色反应，但是否足以明确“school literature clubroom”有边界性；正向理由把任务开头原句/诗歌点评直接当作场景已被清楚传达，可能高估地点证据。

### #3 — reasonable — `exp_236d2505e5f94045a97a25e503841cf5`

子经验：`exp_cd0dcf68a13b4672950233a0b5e8b804`、`exp_49810b5412314906a722bf90ea7979b6`。

两组都要求严谨，但所需方法有实质差异：旧案例是信息不足时校准推断、说明局限并给出可支持的解释；TREE(3) 案例要求按数学对象的定义判断大小关系，不能把对象误读成最大值/无穷。定义核验不能由一般“承认不确定”替代，独立经验有检索价值。

- `helpsteer3:train:62080c0645bcc1765f028b188afbd1cbeca226e2ae7f3f88d2a2108508b48a26`：Task 问莎士比亚句中“my”的含义；chosen 说明缺上下文时不能确定具体原因，并解释基础语义。
- `helpsteer3:train:fdae9b676f0207d2bf88682a814f30585f189dbdf66bb45149a87a6508932d2b`：Task 问“best” fingerstyle instructors；chosen 说明最佳选择取决于学习风格、目标和偏好。
- `helpsteer3:train:6bd4a8d33c9d46468ca1fb455b5e0ba6de60a0e8023554f8509fa59273033b95`：Task 问是否有数大于 TREE(3)；chosen 用 TREE(4) 作例并说明 TREE(3) 不是最大数。

可兼容统一经验：可上提为“先辨明问题中的概念与信息边界：信息不足时说明限制并给可支持解释；数学对象则按定义比较，不把巨大误作最大或无限。”不过这会把两个不同的具体评估动作压在一条经验里。

案例/Judge/漂移：当前样本支持“存在更大的数”这一结论，chosen 的 TREE(4)>TREE(3) 例子与问题相符；它并未实际犯“最大/无限”错误，因此这次信用只说明该标准适用并获正向判定，不能证明新数学经验纠正了真实错误。

### #4 — reasonable — `exp_71dc77f6c81544d5a21d9d86af6b036f`

子经验：`exp_1dc8ee2f6e064f89a82ace9ee94c8476`、`exp_f62bd41c2f414ca2b3870c89f109f7b5`。

第一组评估创作文本是否有完整事件结构、结局和人物状态演变；第二组评估分析回答是否解释一个表面矛盾及其内在动机。后者不是故事情节是否闭合，无法由前者有效覆盖，属于不同任务与判断机制。

- `helpsteer3:train:2869b9bd2ddd218fd2da467d2ff5abbfb210a96bcfe328d184572113513abc80`：Task 要求“write this story”；chosen 写出遭遇兽群、逃跑、吹笛脱险及结尾。
- `helpsteer3:train:00320069484b7f2aefc5e7f7b79ae08c8c1b1942eaa002be999911d187382dca`：Task 要求完整池塘喜剧故事；chosen 收集鱼、回岸、生火烹鱼，rejected 中途转到旁人发现鱼，未完成指定链条。
- `helpsteer3:train:882b68567238307ec98cb13267dcb4820df4979de90fbdd5c43a94001f58ff3c`：Task 要求解释作者“像是想结束却投入塑造配角”的悖论；Judge 要求解释矛盾双方和内在动机。

可兼容统一经验：不建议合并成同一检索经验：可用一般“先识别任务类型及核心判断目标”作为上位提示，但故事完成度与悖论分析需要不同的判分规则。

案例/Judge/漂移：882 的 chosen 加入了题目没给出的原创作品、人物和情节并声称作者倦怠/内心挣扎；Judge 对“作者动机”是否有文本支持应谨慎，不能把创作性补写当成对原漫画作者真实动机的事实解释。

### #5 — unnecessary — `exp_cd0dcf68a13b4672950233a0b5e8b804`

子经验：`exp_8265e535ba62442c91cb1358b3a55e3c`、`exp_ce06cec953bc48628a0bb2310c067ba8`。

新案要求解释 Big Five 语境下的非标准词并正确映射/区分维度；旧案本来就要求处理模糊定义、非标准术语、说明局限并给有根据的关联解释。两者可由同一术语解释方法覆盖，后者只是把“有关联解释”具体化为不得错误归因。

- `helpsteer3:train:62080c0645bcc1765f028b188afbd1cbeca226e2ae7f3f88d2a2108508b48a26`：Judge 标准要求在缺少上下文时提供“基础语义或逻辑推演”，不把诚实局限判作无内容。
- `helpsteer3:train:fdae9b676f0207d2bf88682a814f30585f189dbdf66bb45149a87a6508932d2b`：chosen 对“best”的含义不作绝对排名，并指出应结合学习者目标和偏好。
- `helpsteer3:train:fee7f87914a5eb136b2325e21b490b590f0bf987f8313d0e9b2bdafb65ac73d4`：Task 同时给出 social、limbic、organized、egocentric、inquisitive；Judge 指出 chosen 把 egocentric 主要归到 neuroticism 是错误的。

可兼容统一经验：遇到非标准或含糊术语时，先说明词义/框架边界，再给证据支持的关联映射；明确区分“信息不足需保留判断”和“已给出术语仍需准确归类”，不要用免责声明掩盖错误归因。

案例/Judge/漂移：fee 样本的 Judge 对 limbic 的“准确映射”给 true，但其理由仅凭回答把该词关联到情绪/记忆并隐含映射，并未证明它是标准 Big Five 维度；egocentric 误映射则被另一标准判 false。正信用来自部分标准，不能据此认定整段术语解释正确。

### #6 — unnecessary — `exp_ccd386af56974af9813c301303e0af1a`

子经验：`exp_ae0844d943eb403cae89705f4f3bf49f`、`exp_baaccc7b82d840d799611ee34b148ef9`。

两组都要求创作回答具备可辨识的场景背景与人物/身体状态；一个案例是单场景对话，一个是三条情境列表。把标准应用到列表时逐项检查即可，属于格式和数量变化，不需另设机制。

- `helpsteer3:train:e3d620cdd3af83d27be22276ca3cc1a5a2f5e74d7e5e51deda8d86ea2d1d503e`：Task 给出“school literature clubroom”及 Monika 点评诗歌场景；criterion 查是否呈现地点和当前情境。
- `helpsteer3:train:ad966f94d1c6f8226d7c7d1686655d76e4d7a946b7e0ab8f471a6659fa45d36a`：Task 要求 create 3 scenarios；Judge 逐条检查地点/环境背景和身体或心理状态是否明确。

可兼容统一经验：创作场景应清楚呈现prompt要求的地点/环境和人物状态；若要求多个情境，对每个情境分别检查，不能以职业标签或概括症状代替场景细节。

案例/Judge/漂移：e3 的 chosen 只用对话和角色反应呈现文学社活动，并通过“literature club”“go around the room”暗示地点；Judge 将其判为学校文学社教室清晰呈现，证据偏宽。ad966 chosen 的三条分别有时尚活动、婚礼、比赛及疼痛细节，正向判定有明确文本支持。Updater 曾因样本分配校验失败重试（validation_feedback），最终分组完整。

### #7 — unnecessary — `exp_ae0844d943eb403cae89705f4f3bf49f`

子经验：`exp_965cdfbc2a184394919880d9a597206f`、`exp_d893391f51fa41ab93d1f629b4717190`。

两组都是对长篇创作 prompt 做要素覆盖核对：旧案检地点/背景，新案检多步骤情节中的具体名词、动作和修饰词。可作为一份按任务列出“场景信息＋必需事件＋关键细节”的覆盖清单，触发条件不同但方法相同。

- `helpsteer3:train:e3d620cdd3af83d27be22276ca3cc1a5a2f5e74d7e5e51deda8d86ea2d1d503e`：criterion：“背景（学校文学社教室）及当前情境”；rejected 未有地点/情境描写。
- `helpsteer3:train:ecfef24beaac45487430ae5482888f4c5bbb97b7fee446ddcdf6350bd0a2f069`：Task 明确指定 miso soup with green onions、rice、onigiri with umeboshi、短裙等一连串细节；chosen 覆盖其中多数，rejected 漏了若干具体细节。

可兼容统一经验：面对包含多项具体要求的创作任务，先拆成场景/背景、事件、角色动作和关键物件或修饰细节逐项比对；遗漏核心要素才判缺失，避免把一项覆盖误当全篇完成。

案例/Judge/漂移：ecfef 的第三项把“beautiful short dress”作为区别 rejected 的关键形容词；rejected 确实没有“short/beautiful dress”完整信息，但“beautiful”是否核心、是否应按词面核对过于严格。Chosen 也只把饮食等放在概述中，Judge 的合格判断应关注信息是否出现而非要求原词复现。

### #8 — unnecessary — `exp_764647345b1845a79a07a098387fbe8f`

子经验：`exp_40c8ecbcfbd746029df1dcbae253af4c`、`exp_2fca6add4b3f455d9c5300411d6cb1a2`。

旧经验处理实体不存在/歧义时先澄清替代；新经验处理跨作品对比时查实人物真实出处、状态和属性。两者是实体消歧与事实落地的同一方法分支，可用“先确认实体身份及来源，再决定澄清还是按可证事实作答”覆盖。

- `helpsteer3:train:fa4ca7a2821fe18e1b24549bec186d600efbf248bede61bdabf22a0b47b40ef3`：Task “Blu and Lola Loud Character”；chosen 先说 Blu 不是 The Loud House 官方角色并提供可能替代 Lincoln。
- `helpsteer3:train:0919d7efcd6dcf730de2ce09a16c238a3be30528a8d6199403981c7397ee6e96`：Task 明示 Tommy Oliver 来自 Power Rangers comics、Brandon Carver 来自 TWD TV Series；标准检是否保持来源，不把后者移植为另一系列角色。

可兼容统一经验：涉及冷门、歧义或跨作品实体时，先确认名称、作品来源和存在状态；不确定则说明局限并求澄清/给替代，确认后只陈述可由该来源支持的属性，不把相似角色资料移植过来。

案例/Judge/漂移：0919 chosen 虽复述任务给定的 TWD 来源，但具体能力只泛称为“skilled survivor/capable fighter”；positive criteria 还把这些无细节支撑的描述视作基于剧集常识。该正向判断没有证明其角色属性确实正确；rejected 的来源混淆则被正确识别。

### #9 — unnecessary — `exp_965cdfbc2a184394919880d9a597206f`

子经验：`exp_77fe3f7fcf4746098b348a26408564f6`、`exp_ecc9357b2aab4b79a00d32958480e2fc`。

与拆分7属于同一创作任务覆盖方法。新案例比单纯背景更具体，要求核对场景、角色和动作对象/过程；旧经验可扩展为逐项核对prompt指定信息。后续拆分10、14又从同一 e664 样本分离事件链和主客体，显示此分支被重复切细。

- `helpsteer3:train:e3d620cdd3af83d27be22276ca3cc1a5a2f5e74d7e5e51deda8d86ea2d1d503e`：chosen 有 Monika 对 Yuri/Natsuki/Sayori 的点评；被引用criterion检查学校文学社背景是否传达。
- `helpsteer3:train:e6642406424248ec3de42d1b651bc6678bde0b172f317f08e24ca7dd2a0a80eb`：Task 要 Nerd 观看 Bliss 对 P.I. 的比赛；chosen 对应描述录像场景与 Bliss 对 P.I. 的剪刀腿，rejected 改成 Nerd 与 Bliss 的实时对战。

可兼容统一经验：故事生成评分应逐项核对指定场景、参与者、动作对象、过程和结果是否在回答中出现；让每项都有文本证据，避免角色替换或把任务改写成另一场景。

案例/Judge/漂移：e664 的核心错误是 rejected 把观看录像中的 P.I. 换成 Nerd 并改写比赛结局，Judge 清楚指出；chosen 正确保留 P.I.。这与后续拆分14的“主语/对象归属”同一证据链，单独再拆一次有重复索引风险。

### #10 — unnecessary — `exp_d893391f51fa41ab93d1f629b4717190`

子经验：`exp_b9bacc420eca44f7b943cfb791f14624`、`exp_230394ac195b4fcb93739de01dac5551`。

“关键形容词/名词短语”与“特定事件链、角色动作、生理变化”都是对同一长prompt的要素覆盖检查；后一组列出更大粒度要素，前一组检查细节精度。可以一条经验同时要求先确保事件链和角色归属，再核对限定性细节。

- `helpsteer3:train:ecfef24beaac45487430ae5482888f4c5bbb97b7fee446ddcdf6350bd0a2f069`：Task 要求 waking breakfast、上学、被监视、摸额头、购物对话和回忆等多个步骤与细节。
- `helpsteer3:train:e6642406424248ec3de42d1b651bc6678bde0b172f317f08e24ca7dd2a0a80eb`：Task 指定观看 Bliss 对 P.I. 的比赛、长时间剪刀腿、P.I. 投降后仍被挤压至昏迷。

可兼容统一经验：按prompt建立覆盖清单，逐项核对事件顺序、参与角色/动作对象、状态变化和关键限定词；同义表达可接受，但不能遗漏改变任务含义的角色、动作或结果。

案例/Judge/漂移：Judge 对 e664 rejected 指出角色及事件链错置正确；chosen 对 P.I. 投降、继续施压至昏迷有清晰证据。Chosen 未字面使用“shapely”，但描述“thighs a vice of flesh and muscle”，Judge 将其视为等效表达是合理的。

### #11 — unnecessary — `exp_40c8ecbcfbd746029df1dcbae253af4c`

子经验：`exp_a026765e795b4c0190f831204c860f35`、`exp_a15d2a3b70af4b87a4b1474f562236bc`。

前案要求对不存在/模糊实体说明状态并避免虚构；后案要求复述影视情节时严格依照原作、不添加不实细节。两者共享事实来源约束，差别是未知实体与熟悉作品的上下文，不构成互斥机制。

- `helpsteer3:train:fa4ca7a2821fe18e1b24549bec186d600efbf248bede61bdabf22a0b47b40ef3`：Chosen 先指出 Blu 不是该作品官方角色，再给可能替代；rejected 虚构其履历、口头语和关系。
- `helpsteer3:train:6611eb8a54b4339c881dc41fbc203a1033c24c8a516827ff209e54f1f8ca6ba5`：Task 问 Shutter Island 的最佳解释；Judge 理由称 rejected 编造岛名来源并错写 Rachel 消失地点（题内未提供独立原作证据）。

可兼容统一经验：回答作品/实体问题时只陈述有来源支撑的事实；若身份或细节不确定，先澄清或声明限制，不为填补空白编造履历、情节或背景。

案例/Judge/漂移：6611 chosen 看起来符合题目给的电影语境；Judge 对 rejected 的“Shutter”命名缘由及 Rachel 消失位置提出具体事实异议，但本审查未外查原片/原著，题内证据无法独立确认这些断言。这里只记录 Judge 提出了来源忠实问题，不把其断言当作已核验事实。无明显Judge布尔矛盾。

### #12 — unnecessary — `exp_a026765e795b4c0190f831204c860f35`

子经验：`exp_f05daf8a8afd45f1a3ee60411742dd4d`、`exp_a7554c55d27f4fde869618f1c473db18`。

一组要求面对虚构/模糊实体时不捏造身份特征；另一组要求描述商标视觉属性时以事实为据、按要求呈现“logo＋公司名”组合。都是特定对象属性核验与避免虚构，可共用“先确认对象及属性，逐项按请求呈现”方法。

- `helpsteer3:train:fa4ca7a2821fe18e1b24549bec186d600efbf248bede61bdabf22a0b47b40ef3`：rejected 为不存在的 Blu 编造性格、catchphrase 和 quirk；chosen 作存在性澄清。
- `helpsteer3:train:f10a03360786655db1e4a00f995a312373ceb111c26a1b3c89a5eb5999f8355e`：Task 要求随机公司logo且每项先有“logo”；chosen 写“Logo: Apple”及咬痕等视觉说明，rejected 只列公司与行业。

可兼容统一经验：描述特定实体属性时，先确认对象与事实依据，再按任务格式逐项呈现可支持的视觉/身份特征；无法确认的属性说明不确定，不能虚构。

案例/Judge/漂移：chosen 以文字描述 logo，并未实际展示图像；criterion 将“描述logo与单词的组合形式”作为通过项是按文本任务的合理解释，但如果原意是需要真正的logo图像，Judge没有核实实际视觉呈现。

### #13 — unnecessary — `exp_b9bacc420eca44f7b943cfb791f14624`

子经验：`exp_8f05e8d352c44197b408b7ca009e0228`、`exp_f48fd0d724814f65bea334f204bd50bd`。

两组都在检查指定细节是否真实出现在生成结果中：一组偏名词/形容词，一组偏动作后的结果和状态变化。用一个逐项核对事件、参与者、限定描述和结果的任务覆盖方法即可涵盖。

- `helpsteer3:train:ecfef24beaac45487430ae5482888f4c5bbb97b7fee446ddcdf6350bd0a2f069`：Judge 对 rejected 仅要求 short/beautiful dress 等细节的关键覆盖；chosen 有“short dress with straps”。
- `helpsteer3:train:e5fd06c3a0f2ef1f6e9edd5b941144c138e946776d9d18dec8f60a1f883db90e`：Task 明说 Joji 放屁后明显松弛、鱼翻肚、Joji 开玩笑道歉；chosen 写鱼游开且未写松弛。

可兼容统一经验：核对prompt指定的必需动作、关键限定词和结果/状态变化是否实际出现；接受意义等价的表达，但不能把相似前置动作当成指定结果已发生。

案例/Judge/漂移：P>0 仅来自 c4“jokingly acts sorry”：chosen 的“mock-serious expression / makes a show of apologizing”是语义等价表达，Judge 判 true 合理。visibly relieved 与 fish belly-up 两项 chosen/rejected 都是 false，没有区分力；chosen 确实漏了松弛状态和鱼翻肚，但这不足以支持“仅相似描述不达标”。该输出有把具体状态缺失过度泛化为拒绝语义等价表达的漂移风险。

### #14 — reasonable — `exp_230394ac195b4fcb93739de01dac5551`

子经验：`exp_2ffd017943ab413995de305f7d7e6fd5`、`exp_365113acba21412c8327009e3d96f486`。

这次拆分把一般事件链核对与多角色任务中的施事/受事绑定分开。d18 的错误不是遗漏了“抚摸/脸红/心形瞳孔”这些事件，而是把行为从 Lilac 错接到 Soy-Soy；角色—动作—对象匹配是可独立复用的关系核对机制，能补足 #9/#10 的覆盖清单。

- `helpsteer3:train:e6642406424248ec3de42d1b651bc6678bde0b172f317f08e24ca7dd2a0a80eb`：Task 指定“Bliss held P.I. in ... headscissors”；rejected 将实时被压制者换成 Nerd。
- `helpsteer3:train:d18f0674f5609ea960ac85fbe949afe1778203d11719f55ac5e95f634ec304cd`：Task 要 Goku pet Lilac、Lilac 脸红并出现心形瞳孔；rejected 实际抚摸和状态变化都归给 Soy-Soy。

可兼容统一经验：在逐项覆盖清单中增加显式关系核对：每个指定动作都确认执行者、动作、对象及状态变化对象相匹配；不可因同一场景中“有人做了该动作”就判满足。

案例/Judge/漂移：d18 的 chosen 明确写 Goku pet Lilac、Lilac blushes、pupils turn into hearts；rejected 把目标错换为 Soy-Soy，Judge 的反事实区分充分。Chosen 其余“IQ 不适用”解释没有相关正向判定，不影响这次关系机制证据。

## 典型例子

- **独立机制值得保留（#4）**：完整故事应检查叙事链与结尾；悖论分析则要解释表面矛盾及内在动机，标准形式不同。
- **细粒度覆盖重复（#7、#10、#13）**：三次都在拆分复杂生成任务的必需信息，依次落到具体描述、事件链、状态/结果。统一成按角色、动作、限定词和结果逐项核对的清单更连贯。
- **真实关系错误（#14）**：d18 的 rejected 虽然写了抚摸和心形瞳孔，却把对象从 Lilac 换成 Soy-Soy；专门核对施事/受事有独立价值。
- **证据噪声（#1）**：c7 的复合标准Judge同时给 chosen/rejected=true，且理由内部改判；该局部正信用不应被当成经验正确性的证明。

范围限制：本审查只评价这14次拆分的语义必要性和案例/Judge证据质量；不据此推断拆分对下游表现的因果效果。
