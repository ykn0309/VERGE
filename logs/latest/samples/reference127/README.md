# 逐题轨迹索引

训练经验路独立试用；冻结测试按推理协议只运行选定路径。

| 编号 | 阶段 | 样本ID | 完整轨迹 | 初始路径结果 | Critic轮数 |
|---|---|---|---|---|---:|
| 1 | train | helpsteer3:train:a2f015e21a54072931237a7e65ba4cbb14379c709331935bce87314916e1b35c | [train-001.json](train-001.json) | 无经验:错 | 2 |
| 2 | train | helpsteer3:train:eb1dd5193e24cea783ce4b95b167ce8dd3f745262bfc2c7ffaeeafc73cf2e095 | [train-002.json](train-002.json) | 无经验:错 | 1 |
| 3 | train | helpsteer3:train:868649a89ffeda08431889b471d17d0d728f477457935546418fd549255a3bac | [train-003.json](train-003.json) | 无经验:对 | 0 |
| 4 | train | helpsteer3:train:3335f8423a397fba5d62cc19099883c785db8b4b7d9e4761ab6dec9760f75dca | [train-004.json](train-004.json) | 无经验:错 | 2 |
| 5 | train | helpsteer3:train:eed9e00d41b8537f9bf32d716a0d1cbb5b8399dc4c862064dd757d60c163d9f0 | [train-005.json](train-005.json) | 无经验:错 | 2 |
| 6 | train | helpsteer3:train:37b19bc9e31261fc7d5d1caeac33ae2a2758ecfb0108a9f334a62ee8bdb03ff5 | [train-006.json](train-006.json) | 无经验:错 | 1 |
| 7 | train | helpsteer3:train:ec6ace7446b9225f25a5c3466091db6b0f032ef8625ae56b8668de3bf4d993b9 | [train-007.json](train-007.json) | 无经验:错；exp_09eb932d6d8240b489b819c8bb9306b2:错 | 2 |
| 8 | train | helpsteer3:train:63e48b1e4bf5455341e4dcbec7ff4e80400bed77605f6e1b460ac4e27dd1998a | [train-008.json](train-008.json) | 无经验:错；exp_09eb932d6d8240b489b819c8bb9306b2:对 | 0 |
| 9 | train | helpsteer3:train:bce814b2202bf09bfd4efc8d614a676d1452694d701b627e4f2f27c0dd2717f2 | [train-009.json](train-009.json) | 无经验:错 | 2 |
| 10 | train | helpsteer3:train:cdcf84155f1b15ee05c5ac68d90e1d8dbf0b7d1a54709987ecf8ea89bf657572 | [train-010.json](train-010.json) | 无经验:错 | 1 |
| 11 | train | helpsteer3:train:d74120094d1c999aedaea7fb09390ceef8c8f3a512fa1e916d24962e0aa4659b | [train-011.json](train-011.json) | 无经验:对；exp_e6c897802cef421ba20109560cd6b0b2:错 | 0 |
| 12 | train | helpsteer3:train:237a91b1f833437d5832ce71741036a6112e8ae1d2114be48e9a2436f21a0e89 | [train-012.json](train-012.json) | 无经验:错；exp_09eb932d6d8240b489b819c8bb9306b2:错 | 2 |
| 13 | train | helpsteer3:train:cd6ad253ae74118cd9a196d1de8aa62b9f8e852a2e891db6cb40ebdf0061c03f | [train-013.json](train-013.json) | 无经验:错；exp_09eb932d6d8240b489b819c8bb9306b2:错 | 1 |
| 14 | train | helpsteer3:train:f349d8a2e1e0f47891b3d62c79aba2408b62563d7bdaa7a3362b68a1ad3f28e1 | [train-014.json](train-014.json) | 无经验:对 | 0 |
| 15 | train | helpsteer3:train:2c766a624bd9bdf29eb1c256593bb29f60a70e5d19f9199f38555d887bffdc89 | [train-015.json](train-015.json) | 无经验:对 | 0 |
| 16 | train | helpsteer3:train:25dcf9463e69637b0d580034fa778f29d2a75bd0325ace5da099d6bdffde2833 | [train-016.json](train-016.json) | 无经验:错；exp_5d494f64f5f34a659522d57b1c811779:对 | 0 |
| 17 | train | helpsteer3:train:b113f9a608f3aedf57e16e136fbb76d412e45d1f7a4461ed0c962cce8cbb52c7 | [train-017.json](train-017.json) | 无经验:错；exp_09eb932d6d8240b489b819c8bb9306b2:错 | 1 |
| 18 | train | helpsteer3:train:3563d0c339932906ab7778697967d6d5554438e08ea753fdfe53a4931e92d97c | [train-018.json](train-018.json) | 无经验:错；exp_09eb932d6d8240b489b819c8bb9306b2:错 | 2 |
| 19 | train | helpsteer3:train:5a7e3514ba9a8466a1120d40ec9a6cfde5aea7bb33fce3daba153352399d1ca4 | [train-019.json](train-019.json) | 无经验:对；exp_09eb932d6d8240b489b819c8bb9306b2:对 | 0 |
| 20 | train | helpsteer3:train:a054a2ce6f2c3d8720514325021dba8dbcede7d9edc54f56f7c83b17b6b40b11 | [train-020.json](train-020.json) | 无经验:对；exp_09eb932d6d8240b489b819c8bb9306b2:对 | 0 |
| 21 | test | helpsteer3:train:e87142c0c0423c2170d028a1348233cec9bf7f9d6be7cb1f44ccbeb88ec302b9 | [test-001.json](test-001.json) | exp_5d494f64f5f34a659522d57b1c811779:错 | 0 |
| 22 | test | helpsteer3:train:d6be25ef49d2e05dc5543a3e9b909e66ccfc6d14130bb533ca44696f0658808c | [test-002.json](test-002.json) | exp_09eb932d6d8240b489b819c8bb9306b2:错 | 0 |
| 23 | test | helpsteer3:train:2a74a498f18192b5e3aa6b3ad27ea274264b0655f2ceea67e6e183ddd54e7fed | [test-003.json](test-003.json) | exp_e6c897802cef421ba20109560cd6b0b2:错 | 0 |
| 24 | test | helpsteer3:train:e52d6fe67a1198fa140dbe63dcc0bae4cb329034367944f16e346666e5828ba3 | [test-004.json](test-004.json) | exp_88444417d4724563aeb575ab56bd2c72:错 | 0 |
| 25 | test | helpsteer3:train:f2793dcabf46cc9c0dfa34f4ae7811e83fdaca80a3ae699fab3122556d6a8aa6 | [test-005.json](test-005.json) | exp_09eb932d6d8240b489b819c8bb9306b2:错 | 0 |
