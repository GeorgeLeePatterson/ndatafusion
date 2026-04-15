# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Miscellaneous Tasks

- Removes rogue docsrs cfg_attr ([91531fc](https://github.com/georgeleepatterson/ndatafusion/commit/91531fc95de5e46a2a78da6aa5e90123822bc25e))
- Patches bug in ci ([a2c962c](https://github.com/georgeleepatterson/ndatafusion/commit/a2c962cf04c2d72407f7ea975ac0849ec381c4a6))
- Addresses dependabot churn ([2ad71f6](https://github.com/georgeleepatterson/ndatafusion/commit/2ad71f6228c00877161018b437b8515d3a108b52))

## [0.1.0] - 2026-04-15

### Bug Fixes

- Addresses f64 scalars and removes unnecessary arrow feature ([7dbee8c](https://github.com/georgeleepatterson/ndatafusion/commit/7dbee8c3237f55a50a47a9204110f82ec8c34a08))
- Add shape constructors for conversions ([93d26b3](https://github.com/georgeleepatterson/ndatafusion/commit/93d26b35a0013fa402f0c034406014cc591768a3))
- Adds matrix functions and triangular solves ([f6020ab](https://github.com/georgeleepatterson/ndatafusion/commit/f6020abc074e950b39a46b64522fb7728e5bea17))
- Introduces additional matrix decomps and new tensor udfs ([1da2725](https://github.com/georgeleepatterson/ndatafusion/commit/1da27251ae755d7e9cc13c2932a8ecae0458a9be))
- Additional matrix functions ([b831726](https://github.com/georgeleepatterson/ndatafusion/commit/b831726c85504a87fdf66ce68d1dda13da5ee44a))
- Addresses remaining non-controversial items, like matrix PCA and remaining sparse ([b34a49e](https://github.com/georgeleepatterson/ndatafusion/commit/b34a49ec94a7906ee1f1f8c1c2f418e77b84e252))
- Preparing for initial release ([011a941](https://github.com/georgeleepatterson/ndatafusion/commit/011a9410dddd29adf830f997e999138a0246d332))
- Adds aliases and naming conventions ([9eabc89](https://github.com/georgeleepatterson/ndatafusion/commit/9eabc8994911c002575c6dec358badc75c417268))
- Implements some table and window UDFs ([492f9bb](https://github.com/georgeleepatterson/ndatafusion/commit/492f9bb5c4a173370ffe46ecee449dce53532969))
- Addresses security audit ([bb43d6d](https://github.com/georgeleepatterson/ndatafusion/commit/bb43d6de12bfa20b6bde65ff5c20f2562798f167))

### Documentation

- Release preparation and doc updates ([90368ea](https://github.com/georgeleepatterson/ndatafusion/commit/90368ea907ff6ff0411627d85771b08bb6462457))

### Features

- Introduces UDAFs, adds named parameters, improves documentation ([88c6350](https://github.com/georgeleepatterson/ndatafusion/commit/88c6350dad733fdeeb74299c79d397909829fc47))
- Addresses remaining actionable udafs ([15650d3](https://github.com/georgeleepatterson/ndatafusion/commit/15650d31807a4a38ca26f08b511d1ee6ccaa0d23))
- Introduces complex matrix surface ([ddac1fc](https://github.com/georgeleepatterson/ndatafusion/commit/ddac1fc6e2a092707e98ee4cef43c298f7a12a3c))
- Introduces additional complex decomps and other functions ([d276cfa](https://github.com/georgeleepatterson/ndatafusion/commit/d276cfa980774f983819a0cfaf2964a86c34ffb1))
- Introduces additional complex PCA ([179d5cb](https://github.com/georgeleepatterson/ndatafusion/commit/179d5cbc8caad595d425ebcf44d3829c5b65e8b8))

### Miscellaneous Tasks

- Initializes AGENTS.md, points datafusion to a rev on main ([06402ba](https://github.com/georgeleepatterson/ndatafusion/commit/06402ba935a51aaab104b58d7ebd996d61b0bec4))
- Updates branch name ([c644b8e](https://github.com/georgeleepatterson/ndatafusion/commit/c644b8e7867f5759b042b678b5a6cd68d87d3f26))
- Patches missing rust component in ci ([e2a80a3](https://github.com/georgeleepatterson/ndatafusion/commit/e2a80a3dea0f2d24b2ea630fa367e6090d581429))
- Another patch to ci ([a38a8fa](https://github.com/georgeleepatterson/ndatafusion/commit/a38a8fa591ae4977c2c0a842f424eaf65d14007c))
- Another patch to ci ([395742e](https://github.com/georgeleepatterson/ndatafusion/commit/395742e4319d96357a00981d78f792eec5b69e23))
- Patches cov in ci ([814d1fb](https://github.com/georgeleepatterson/ndatafusion/commit/814d1fb72812aa66e1752416583939cb502a16cc))
- Renamed udf test suite due to exclusion by llvm-cov ([331c46a](https://github.com/georgeleepatterson/ndatafusion/commit/331c46ab0909c5c69e03075ecc99354f7bc4fe29))
- Cargo.toml reformatting ([ddff72e](https://github.com/georgeleepatterson/ndatafusion/commit/ddff72e6612e47395d9af677687c44272a13484b))
- Updates workflows ([32c85b1](https://github.com/georgeleepatterson/ndatafusion/commit/32c85b1f385ebc1e8d7ebb124fbefa6bb4de1401))
- Updates release workflow ([30080a6](https://github.com/georgeleepatterson/ndatafusion/commit/30080a6b2e31792ba48d8fd7364e6a65d07c1092))
- Updates ci for incremental builds ([ec6133e](https://github.com/georgeleepatterson/ndatafusion/commit/ec6133e9461b3ab176acafcefda220943c6b53d9))

### Refactor

- Normalizes to an f64/f32 surface area ([3efbd14](https://github.com/georgeleepatterson/ndatafusion/commit/3efbd14c7650dfe8d5314a16948d0aa48b478926))

### Testing

- Addresses test coverage and lint miss ([8524df7](https://github.com/georgeleepatterson/ndatafusion/commit/8524df7cba83fe877e99dc3188e0bd5dc075d50b))

### Example

- Added examples ([064f024](https://github.com/georgeleepatterson/ndatafusion/commit/064f024574c372ac63294e79efe10841253e6301))


