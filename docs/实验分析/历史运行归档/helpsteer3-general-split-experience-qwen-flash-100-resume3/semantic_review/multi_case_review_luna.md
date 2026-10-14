# Luna 对 6 条活跃多案例经验的逐案例核验（修订）

审查证据源：`multi_case_evidence.json`。共核对 **18条经验归属、14个不同 sample_id**；4个 sample_id 各归属到两条经验。以下将案例是否体现相关现象，与其是否验证当前经验的完整方法规则分开标记。正信用/Judge margin不是经验有效性的证据。未进行外部事实查询；题内无法确认的事实保留不确定。

**判定口径：** `phenomenon_supported=true` 表示案例出现相关现象或提供正确配对/正例；`full_method_supported=true` 才表示案例直接验证了当前规则的关键判断。完整经验的总体“支持”要求核心子句都有直接证据。

## `exp_f48fd0d724814f65bea334f204bd50bd` — 部分支持
当前文本：针对prompt指定的生理状态变化（如脸红、瞳孔变形）或具体动作后果，必须显式复述其核心特征词，相似描述不达标。

- **部分支持** `helpsteer3:train:e5fd06c3a0f2ef1f6e9edd5b941144c138e946776d9d18dec8f60a1f883db90e`；现象支持：是；完整方法规则支持：否。chosen对“visibly relieved”和“fish go belly-up”均未满足，Judge的c2/c3都是chosen=false、rejected=false：识别了状态/结果遗漏，但两项没有区分两回答，不能解释正信用。唯一P=0.1来自c4：chosen写“puts on a mock-serious expression and makes a show of apologizing to the fish”，Judge判为符合“jokingly acts sorry”，给chosen=true、rejected=false。该语义改写的正例说明无需照抄原词；它支持覆盖任务语义，不支持强制词面。 英文证据：“visibly relieved”；“fish go belly-up”；“puts on a mock-serious expression and makes a show of apologizing to the fish”
  Judge/标准问题：c2/c3两侧皆false，只识别遗漏而不贡献区分；c4的语义改写道歉获得P=0.1，正好不支持照抄原词。任务语义要求不等于字面词要求。
- **部分支持** `helpsteer3:train:d18f0674f5609ea960ac85fbe949afe1778203d11719f55ac5e95f634ec304cd`；现象支持：是；完整方法规则支持：否。支持动作对象和生理状态正确归属；chosen用“blushes”“pupils turn into hearts”对应prompt的“blushed”“heart-shaped pupils”，属于语义匹配正例，没有证明同义改写会被拒绝或必须照抄核心词。 英文证据：“decides to pet her”；“blushes and her pupils turn into hearts”；“Goku pets Soy-Soy”
  Judge/标准问题：相关对象/状态判断准确；“31 iq/非英语”等模糊附加要求不影响此结论。

## `exp_5c518dac2c5b49d780627349e7b723e6` — 部分支持
当前文本：核验故事是否具备起承转合结构及主角状态演变；仅当缺失核心逻辑（如缺结局、情节中断）或仅复制标题时判未满足。

- **支持** `helpsteer3:train:2869b9bd2ddd218fd2da467d2ff5abbfb210a96bcfe328d184572113513abc80`；现象支持：是；完整方法规则支持：是。chosen交代男人遇兽群、被追、吹笛脱险并离开；rejected原样复制标题。直接支持仅复制标题或缺结局时才判未满足，但不能单独证明更强的叙事质量标准。 英文证据：“Ethan took off running”；“carefully made his way out of the forest”；“Write this story: a man finds himself running…”
  Judge/标准问题：Judge把重复标题评为无故事内容准确；当前经验已从历史的背景分散问题泛化。
- **部分支持** `helpsteer3:train:00320069484b7f2aefc5e7f7b79ae08c8c1b1942eaa002be999911d187382dca`；现象支持：是；完整方法规则支持：否。chosen完成放松、放屁、收鱼、烹饪结局；rejected虽有故事结尾，却遗漏任务指定的鱼、篝火和晚餐。支持任务关键结局/核心逻辑缺失；“起承转合”与题设遵循混在rubric中，无法验证一般结构门槛。 英文证据：“With the fish in hand, they built a campfire”；“未进行收集鱼、生火或烹饪”
  Judge/标准问题：Judge对具体剧情缺失有据，但以结构缺陷解释混入了题设遵循。
- **部分支持** `helpsteer3:train:e6642406424248ec3de42d1b651bc6678bde0b172f317f08e24ca7dd2a0a80eb`；现象支持：是；完整方法规则支持：否。chosen按题设写Nerd看录像、分析并准备；rejected改成现场对打。chosen的状态变化更切题，但两者都有事件发展及结局，主要差异是题设遵循而非一般结构完整性。 英文证据：“He rewound the footage”；“Nerd ... emerged victorious”
  Judge/标准问题：Judge以指定观看情节作为结构标准；不足以说明rejected缺少一般起承转合。
- **部分支持** `helpsteer3:train:c5f644a17ea3dbb927c39153ca1aad0845fdd1ea5d29f21432121b8f9605f51b`；现象支持：是；完整方法规则支持：否。两脚本均有开场、交锋和胜负结局；chosen稍清楚表现Leonardo变强到获胜，rejected也按回合推进并结束。以“缺乏叙事深度/危机转折”否定rejected，超过当前“核心逻辑缺失才否定”的低门槛。 英文证据：“When the dust settles, Leonardo stands victorious”；“Final Round: Decisive Victory”
  Judge/标准问题：Judge的拒绝理由与当前经验的低门槛不一致；最多支持可辨状态/结局。

## `exp_b9705388e27541f2bdec6b78ff8f02de` — 部分支持
当前文本：任务含特定场景/视角时，标准须显式验证环境构建（地点、氛围）及动作过程细节，区分第一人称互动与第三人称旁白。

- **部分支持** `helpsteer3:train:e6642406424248ec3de42d1b651bc6678bde0b172f317f08e24ca7dd2a0a80eb`；现象支持：是；完整方法规则支持：否。chosen建构观看录像的后室、擂台氛围和头剪受力；rejected也有arena/crowd及动作，但把任务改成现场对打。支持具体场景/动作要可见；“第一人称互动与第三人称旁白”的区分未被隔离，两者都用第三人称叙述并有对话。 英文证据：“dimly lit backroom”；“Her leg hooked around P.I.’s neck”；“the crowd roared”
  Judge/标准问题：Judge抓到场景/对象错位，但rejected并非完全没有环境描写；视角子句证据不足。
- **支持** `helpsteer3:train:164b56508699a900f855e6a0af0f429db05b9cf8c158c5b0a37e9480f2fb05d2`；现象支持：是；完整方法规则支持：是。chosen把光线、镜头互动、指责动作写成现场promo；rejected是面向消费者的广告文案，没有角色实际训斥表演。支持环境、动作过程与叙述立场须匹配体裁。 英文证据：“a single spotlight illuminating his face”；“pointing directly at the viewer”；“Look no further!”
  Judge/标准问题：Judge对场景/动作项的差异有据；两侧都提到盗窃与artwork，不是经验主证据。

## `exp_2c25a0e970014475bd32329ba349377c` — 部分支持
当前文本：面对信息缺失、主观模糊或术语歧义，需声明局限性并提供力所能及的客观解释，避免无依据断言。

- **支持** `helpsteer3:train:62080c0645bcc1765f028b188afbd1cbeca226e2ae7f3f88d2a2108508b48a26`；现象支持：是；完整方法规则支持：是。题目只问句中“my”的含义。chosen给出物主代词解释、结合ashamed，并说明上下文不足以确定原因；rejected扩写成积极的“ownership and identity”，与羞耻语气冲突。直接支持声明限制、客观解释、避免无据推断。 英文证据：“possessive pronoun”；“without additional context”；“ownership and identity”
  Judge/标准问题：Judge把未回答深层情感含义的rubric项判两侧不满足合理；题面不足以断言具体家庭背景。
- **部分支持** `helpsteer3:train:fdae9b676f0207d2bf88682a814f30585f189dbdf66bb45149a87a6508932d2b`；现象支持：是；完整方法规则支持：否。“best instructors”有主观性；chosen说适配取决于学习风格、目标和偏好，rejected列出可疑具体课程名且无标准说明。支持限定主观最佳判断；教师/课程真实性需外查，本次不核实。 英文证据：“best instructor for you will depend on your learning style, goals, and preferences”；“The Ultimate Guitar Course”
  Judge/标准问题：Judge对主观适配的判断合理；外部课程事实未核验，不能用来支持经验其余部分。
- **部分支持** `helpsteer3:train:0919d7efcd6dcf730de2ce09a16c238a3be30528a8d6199403981c7397ee6e96`；现象支持：是；完整方法规则支持：否。chosen沿用题面给出的TWD出处，但进一步断言未由题面支持的武器和生存能力；rejected把Brandon改成Power Rangers角色。支持避免错配来源；未明示不确定性且人物细节未经本次外查，不能算完整验证当前经验。 英文证据：“a survivor from the television show The Walking Dead”；“Green Time Force Ranger”
  Judge/标准问题：Judge的rubric/judgment出现“泰坦尼克号”错位；角色外部存在性和剧情本次不查，保留不确定。
- **部分支持** `helpsteer3:train:f0c091b3ff8cba54055740e19cbe247dc22b21acca6578a8647052899bf2edc3`；现象支持：是；完整方法规则支持：否。语境使TMNT解释很可能成立，但“198u”仍含歧义。chosen按TMNT回答并说明优劣主观；rejected转成Imagine Dragons专辑。支持避免实体错配；chosen未处理拼写/解释假设，只支持部分子句。 英文证据：“best TMNT series is subjective”；“198u”；“Imagine Dragons”
  Judge/标准问题：Judge指出未澄清末项有据；作品出处需外查的部分不据此作事实断言。

## `exp_e3203f116ce34f61b1ad4ac8ee71d1c6` — 支持
当前文本：核对标准须精准匹配主语与动作对象，区分不同角色的行为归属及心理/语言活动，避免张冠李戴。

- **支持** `helpsteer3:train:d18f0674f5609ea960ac85fbe949afe1778203d11719f55ac5e95f634ec304cd`；现象支持：是；完整方法规则支持：是。chosen正确把抚摸及其状态变化绑定Lilac，rejected明确把动作对象移到Soy-Soy；这是区分正确配对与角色错配的直接对照。 英文证据：“decides to pet her”；“Lilac ... blushes and her pupils turn into hearts”；“Goku pets Soy-Soy”
  Judge/标准问题：相关归属判断准确；IQ、动物叫声等额外模糊项不影响结论。
- **部分支持** `helpsteer3:train:4b1c23072cfdc18dc8c695b0e63553efbe0c1a1e1b79ccfee4220510c2b4af5e`；现象支持：是；完整方法规则支持：否。chosen把任务给出的11人逐一配对了身份/职业，证明此类任务适用逐项核对；rejected只是拒答，没有出现职业张冠李戴，故不能证明该方法能识别或纠正错配。 英文证据：“Isabella Torres [ARTIST]”；“Emily Nakamura [ARTIST]”；“Kevin Alvarez [DANCE INSTRUCTOR]”
  Judge/标准问题：Rubric称“共10人”却列11人；Judge误称chosen只有9人且遗漏Kevin，和答案矛盾。C1结论错误，但C2配对证据仍有效。
- **部分支持** `helpsteer3:train:e63a6369be84d90e10f1525b4e7a516e6a8b8c37b5ddd182ce38bb9698100758`；现象支持：是；完整方法规则支持：否。chosen延续了前文角色分工，rejected只复制先前Xanthus台词；这显示任务适用角色归属核验，但差异是续写完整性，不是错配。 英文证据：“Xanthus: “Yes, Mom…””；“Sayori…asks”；“Natsuki…explains”
  Judge/标准问题：Judge称chosen缺少性别问题，但它在用户给出的前文中；要求续写重复前文不当，不改变归属证据。
- **支持** `helpsteer3:train:a5346618b26d91d70b749885e9419692348c412200757a3b7fb7fb40235c5969`；现象支持：是；完整方法规则支持：是。chosen保留并区分Goku括号内的内心想法和口头回应，rejected把它们改成普通礼貌台词；直接检验心理/语言活动的归属和类别。 英文证据：“Even that smile creeps me out”；“Will do!”；“Goku, being the kind and respectful person he is”
  Judge/标准问题：Judge关于两类话语区分准确；缺少原台词首个左括号是字面复现问题，不推翻角色归属。

## `exp_d4ee1407c2f245a793e1adfa1ead5c96` — 部分支持
当前文本：跨作品/实体对比时，须先核实出处与存在性；若存疑或虚构，应声明局限性而非强加属性。

- **部分支持** `helpsteer3:train:0919d7efcd6dcf730de2ce09a16c238a3be30528a8d6199403981c7397ee6e96`；现象支持：是；完整方法规则支持：否。chosen至少按题面将Brandon关联TWD；rejected错误改成Power Rangers。支持避免错误作品属性；chosen仍对武器/生存能力作无题内依据的断言，且存在性未外查，故未证实“先核实并声明局限”整句。 英文证据：“a survivor from the television show The Walking Dead”；“Green Time Force Ranger”；“experience using various types of weapons”
  Judge/标准问题：Rubric提到“泰坦尼克号”明显错位，Judge又把无明示限制解读成隐含严谨；外部事实保留不确定。
- **部分支持** `helpsteer3:train:f0c091b3ff8cba54055740e19cbe247dc22b21acca6578a8647052899bf2edc3`；现象支持：是；完整方法规则支持：否。chosen按TMNT语境理解；rejected转为Imagine Dragons并强加专辑/歌曲属性。支持避免错配到非目标作品；chosen没澄清“198u”或说明解释假设，未验证完整的不确定性处理。 英文证据：“Rise, 2012, or 198u”；“best TMNT series is subjective”；“Fast Forward is a great album”
  Judge/标准问题：Judge对错误套入专辑方向有文本依据；外部作品事实本次不查，不将正margin当作实体核实。

## 汇总

18条经验归属的逐案例判定：**支持5、部分支持13、不支持0、不确定0**。其中 18 案出现相关现象或正例，5 案直接支持当前方法规则。14个不同任务里有4个sample_id重复归属，不能把18条归属称作18个独立任务。

关键修订：exp_f48fd整体降为部分支持。Joji任务里所需“visibly relieved”与“fish go belly-up”在chosen中均缺失，Judge两侧皆false；credit来自“mock-serious expression / makes a show of apologizing”，这恰是语义改写并非字面复述。因此证据只支持保留任务语义要求，不支持必须复用prompt核心词。Lilac案例也只有状态及对象正确对应的正例，没有同义复述对照。

exp_e320中4b1与e63标为部分支持：前者chosen职业配对正确但rejected只是拒答；后者chosen正确续写、rejected只复制前文。它们支持适用性/正确配对现象，不证明经验能区分或纠正角色错配。d18提供动作对象错配的直接对照，a534直接检验内心活动与口头表达的区分。

实际模型身份：GPT-6（Codex审查代理），不是Qwen3.5-Flash实验角色。
