# Git 高阶使用与实战场景指南

## 目录
- [0. Git 常用命令速查与应用场景](#0-git-常用命令速查与应用场景)
- [1. Git 内部原理深入理解](#1-git-内部原理深入理解)
- [2. 高级分支管理策略](#2-高级分支管理策略)
- [3. 交互式暂存与提交优化](#3-交互式暂存与提交优化)
- [4. 高级历史重写技巧](#4-高级历史重写技巧)
- [5. 复杂冲突解决方案](#5-复杂冲突解决方案)
- [6. 高级搜索与调试](#6-高级搜索与调试)
- [7. 子模块与 Subtree 管理](#7-子模块与-subtree-管理)
- [8. 性能优化与仓库维护](#8-性能优化与仓库维护)
- [9. 团队协作高级实践](#9-团队协作高级实践)
- [10. Git Hooks 自动化](#10-git-hooks-自动化)
- [11. 实战场景案例](#11-实战场景案例)

---

## 0. Git 常用命令速查与应用场景

### 0.1 初始化与配置

#### 仓库初始化
```bash
# 初始化新仓库
git init

# 克隆远程仓库
git clone <repository-url>

# 克隆到指定目录
git clone <repository-url> <directory>

# 克隆指定分支
git clone -b <branch-name> <repository-url>
```

**应用场景：** 开始新项目或加入已有项目

#### 全局配置
```bash
# 配置用户名和邮箱
git config --global user.name "Your Name"
git config --global user.email "your.email@example.com"

# 配置默认编辑器
git config --global core.editor "code --wait"  # VS Code
git config --global core.editor "vim"           # Vim

# 配置命令别名
git config --global alias.st status
git config --global alias.co checkout
git config --global alias.br branch
git config --global alias.ci commit
git config --global alias.unstage 'reset HEAD --'
git config --global alias.last 'log -1 HEAD'
git config --global alias.visual 'log --graph --oneline --all'

# 查看所有配置
git config --list

# 查看特定配置
git config user.name
```

**应用场景：** 首次使用Git或更换开发环境

#### 项目级配置
```bash
# 为当前项目配置不同的用户信息（工作项目 vs 个人项目）
git config user.name "Work Name"
git config user.email "work@company.com"

# 配置换行符处理
git config core.autocrlf true   # Windows
git config core.autocrlf input  # macOS/Linux

# 配置忽略文件权限变更
git config core.fileMode false
```

---

### 0.2 日常工作流命令

#### 查看状态与差异
```bash
# 查看工作区状态
git status

# 简洁版状态
git status -s

# 查看未暂存的更改
git diff

# 查看已暂存的更改
git diff --staged
git diff --cached

# 比较工作区与指定提交
git diff <commit>

# 比较两个提交
git diff <commit1> <commit2>

# 比较两个分支
git diff branch1..branch2

# 只显示文件名
git diff --name-only

# 显示统计信息
git diff --stat
```

**应用场景：** 每次提交前检查更改内容

#### 添加与提交
```bash
# 添加指定文件
git add <file>

# 添加所有更改
git add .
git add -A

# 交互式添加
git add -p

# 提交更改
git commit -m "commit message"

# 添加并提交（仅已跟踪文件）
git commit -am "commit message"

# 修改最后一次提交
git commit --amend

# 修改最后一次提交信息
git commit --amend -m "new message"

# 空提交（触发CI/CD）
git commit --allow-empty -m "trigger build"
```

**应用场景：** 保存工作进度

#### 撤销操作
```bash
# 撤销工作区的修改（危险操作）
git checkout -- <file>
git restore <file>  # 新版本推荐

# 取消暂存文件
git reset HEAD <file>
git restore --staged <file>  # 新版本推荐

# 撤销最后一次提交，保留更改
git reset --soft HEAD~1

# 撤销最后一次提交，不保留更改（危险）
git reset --hard HEAD~1

# 撤销指定提交（创建新提交）
git revert <commit>

# 清理未跟踪文件（预览）
git clean -n

# 清理未跟踪文件
git clean -f

# 清理未跟踪文件和目录
git clean -fd
```

**应用场景：** 修正错误或清理工作区

---

### 0.3 分支操作

#### 分支创建与切换
```bash
# 查看所有分支
git branch
git branch -a  # 包括远程分支
git branch -r  # 仅远程分支

# 创建新分支
git branch <branch-name>

# 切换分支
git checkout <branch-name>
git switch <branch-name>  # 新版本推荐

# 创建并切换到新分支
git checkout -b <branch-name>
git switch -c <branch-name>  # 新版本推荐

# 基于远程分支创建本地分支
git checkout -b <branch-name> origin/<branch-name>

# 基于指定提交创建分支
git checkout -b <branch-name> <commit-hash>
```

**应用场景：** 开发新功能或修复Bug

#### 分支合并
```bash
# 合并指定分支到当前分支
git merge <branch-name>

# 不使用快进合并
git merge --no-ff <branch-name>

# 合并时squash所有提交
git merge --squash <branch-name>

# 取消合并
git merge --abort
```

**应用场景：** 整合功能分支到主分支

#### 分支删除与清理
```bash
# 删除本地分支
git branch -d <branch-name>

# 强制删除本地分支
git branch -D <branch-name>

# 删除远程分支
git push origin --delete <branch-name>

# 批量删除已合并的分支
git branch --merged | grep -v "\*\|main\|develop" | xargs -n 1 git branch -d

# 清理远程已删除的本地追踪分支
git fetch --prune
git remote prune origin
```

**应用场景：** 保持仓库整洁

#### 分支重命名
```bash
# 重命名当前分支
git branch -m <new-name>

# 重命名指定分支
git branch -m <old-name> <new-name>

# 删除远程旧分支并推送新分支
git push origin --delete <old-name>
git push origin <new-name>
git push origin -u <new-name>
```

**应用场景：** 修正分支命名错误

---

### 0.4 远程仓库操作

#### 远程仓库管理
```bash
# 查看远程仓库
git remote -v

# 添加远程仓库
git remote add origin <url>

# 修改远程仓库地址
git remote set-url origin <new-url>

# 重命名远程仓库
git remote rename origin upstream

# 删除远程仓库
git remote remove origin

# 查看远程仓库详细信息
git remote show origin
```

**应用场景：** 管理多个远程仓库（如fork的项目）

#### 拉取与推送
```bash
# 拉取远程更新
git fetch origin

# 拉取并合并
git pull origin main

# 使用rebase拉取
git pull --rebase origin main

# 推送到远程
git push origin main

# 推送所有分支
git push origin --all

# 推送标签
git push origin --tags
git push origin <tag-name>

# 强制推送（危险，会覆盖远程历史）
git push -f origin main
git push --force-with-lease origin main  # 更安全的强制推送

# 设置上游分支
git push -u origin main
```

**应用场景：** 同步本地和远程代码

#### 跟踪远程分支
```bash
# 设置当前分支跟踪远程分支
git branch --set-upstream-to=origin/<branch> <local-branch>
git branch -u origin/<branch>

# 查看跟踪关系
git branch -vv

# 拉取所有远程分支
git fetch --all
```

**应用场景：** 团队协作时同步分支

---

### 0.5 历史查看与比较

#### 提交历史
```bash
# 查看提交历史
git log

# 单行显示
git log --oneline

# 显示最近n条
git log -n 5

# 图形化显示分支
git log --graph --oneline --all

# 显示每次提交的文件变更
git log --stat

# 显示每次提交的详细更改
git log -p

# 查看某个文件的历史
git log -- <file>

# 搜索提交信息
git log --grep="keyword"

# 按作者筛选
git log --author="name"

# 按时间筛选
git log --since="2024-01-01"
git log --until="2024-12-31"
git log --since="2 weeks ago"

# 查看某两个提交之间的历史
git log <commit1>..<commit2>

# 美化输出
git log --pretty=format:"%h - %an, %ar : %s"
```

**应用场景：** 代码审查、追踪变更

#### 查看提交详情
```bash
# 查看最新提交
git show

# 查看指定提交
git show <commit-hash>

# 查看某个提交的某个文件
git show <commit-hash>:<file>

# 查看标签详情
git show <tag-name>
```

**应用场景：** 检查特定提交的内容

#### 文件历史追踪
```bash
# 查看文件每行的修改记录
git blame <file>

# 查看指定行范围
git blame -L 10,20 <file>

# 查看文件的修改历史
git log -p <file>

# 查看文件的重命名历史
git log --follow <file>
```

**应用场景：** 追踪Bug来源

---

### 0.6 标签管理

#### 创建标签
```bash
# 创建轻量标签
git tag v1.0.0

# 创建附注标签（推荐）
git tag -a v1.0.0 -m "Release version 1.0.0"

# 为历史提交打标签
git tag -a v0.9.0 <commit-hash> -m "Version 0.9.0"

# 创建签名标签
git tag -s v1.0.0 -m "Signed release"
```

**应用场景：** 标记发布版本

#### 查看与删除标签
```bash
# 列出所有标签
git tag

# 查找特定标签
git tag -l "v1.8.*"

# 查看标签详情
git show v1.0.0

# 删除本地标签
git tag -d v1.0.0

# 删除远程标签
git push origin --delete v1.0.0

# 推送标签到远程
git push origin v1.0.0
git push origin --tags  # 推送所有标签
```

**应用场景：** 版本发布管理

---

### 0.7 暂存工作进度

#### Stash 操作
```bash
# 保存当前工作进度
git stash

# 保存时添加说明
git stash save "work in progress on feature X"

# 包括未跟踪文件
git stash -u

# 包括忽略的文件
git stash -a

# 查看stash列表
git stash list

# 查看stash内容
git stash show
git stash show -p  # 显示详细差异

# 应用最新的stash
git stash apply

# 应用指定的stash
git stash apply stash@{2}

# 应用并删除stash
git stash pop

# 删除stash
git stash drop stash@{0}

# 清空所有stash
git stash clear

# 从stash创建分支
git stash branch <branch-name>
```

**应用场景：** 临时切换任务、保存未完成工作

---

### 0.8 高级搜索

#### 内容搜索
```bash
# 在工作区搜索
git grep "search-term"

# 显示行号
git grep -n "search-term"

# 统计匹配数量
git grep -c "search-term"

# 在特定提交中搜索
git grep "search-term" <commit-hash>

# 在所有分支中搜索
git grep "search-term" $(git rev-list --all)
```

**应用场景：** 代码审计、查找API使用

#### 查找引入Bug的提交
```bash
# 开始二分查找
git bisect start
git bisect bad  # 当前版本有bug
git bisect good <commit-hash>  # 已知的好版本

# 标记当前版本
git bisect good  # 或 git bisect bad

# 自动化bisect
git bisect run <test-script>

# 结束bisect
git bisect reset
```

**应用场景：** 快速定位引入问题的提交

---

### 0.9 工作区管理

#### .gitignore 文件
```bash
# .gitignore 示例
# 忽略所有 .log 文件
*.log

# 忽略 node_modules 目录
node_modules/

# 忽略所有 .env 文件
.env
.env.local

# 但不忽略 .env.example
!.env.example

# 忽略所有 .pdf 文件，除了 docs/ 目录下的
*.pdf
!docs/*.pdf

# 忽略构建产物
dist/
build/
*.min.js
```

#### 已跟踪文件的处理
```bash
# 停止跟踪文件但保留在工作区
git rm --cached <file>

# 停止跟踪目录
git rm -r --cached <directory>

# 更新.gitignore后应用
git rm -r --cached .
git add .
git commit -m "Update .gitignore"
```

**应用场景：** 排除不需要版本控制的文件

---

### 0.10 紧急操作

#### 恢复操作
```bash
# 查看操作历史
git reflog

# 恢复到某个历史状态
git reset --hard HEAD@{2}

# 恢复误删的分支
git checkout -b <branch-name> <commit-hash>

# 恢复误删的文件
git checkout <commit-hash> -- <file>
```

**应用场景：** 误操作后的紧急恢复

#### 临时保存与应用
```bash
# 快速保存所有更改
git stash

# 切换分支处理紧急问题
git checkout hotfix-branch

# 处理完后回到原分支
git checkout original-branch
git stash pop
```

**应用场景：** 紧急处理线上问题

---

### 0.11 协作场景命令组合

#### 场景1：开始新功能开发
```bash
# 1. 更新主分支
git checkout main
git pull origin main

# 2. 创建功能分支
git checkout -b feature/user-login

# 3. 开发并提交
git add .
git commit -m "feat: implement user login"

# 4. 推送到远程
git push -u origin feature/user-login
```

#### 场景2：更新功能分支
```bash
# 1. 保存当前工作
git stash

# 2. 切换到主分支并更新
git checkout main
git pull origin main

# 3. 回到功能分支
git checkout feature/user-login

# 4. 合并主分支的更新
git merge main
# 或使用 rebase
git rebase main

# 5. 恢复工作
git stash pop
```

#### 场景3：代码审查后合并
```bash
# 1. 更新主分支
git checkout main
git pull origin main

# 2. 合并功能分支（不使用快进）
git merge --no-ff feature/user-login

# 3. 推送到远程
git push origin main

# 4. 删除功能分支
git branch -d feature/user-login
git push origin --delete feature/user-login
```

#### 场景4：修复合并冲突
```bash
# 1. 尝试合并
git merge feature-branch

# 2. 查看冲突文件
git status

# 3. 手动解决冲突或使用工具
git mergetool

# 4. 标记为已解决
git add <resolved-file>

# 5. 完成合并
git commit

# 或者放弃合并
git merge --abort
```

#### 场景5：同步Fork的仓库
```bash
# 1. 添加上游仓库
git remote add upstream <original-repo-url>

# 2. 拉取上游更新
git fetch upstream

# 3. 合并到本地主分支
git checkout main
git merge upstream/main

# 4. 推送到自己的远程仓库
git push origin main
```

---

### 0.12 Git 命令速查表

| 类别 | 命令 | 说明 |
|------|------|------|
| **初始化** | `git init` | 初始化仓库 |
| | `git clone <url>` | 克隆仓库 |
| **配置** | `git config --global user.name "name"` | 设置用户名 |
| | `git config --global user.email "email"` | 设置邮箱 |
| **状态** | `git status` | 查看状态 |
| | `git diff` | 查看未暂存的更改 |
| | `git diff --staged` | 查看已暂存的更改 |
| **添加提交** | `git add <file>` | 添加文件 |
| | `git add .` | 添加所有更改 |
| | `git commit -m "msg"` | 提交 |
| | `git commit -am "msg"` | 添加并提交 |
| **分支** | `git branch` | 列出分支 |
| | `git branch <name>` | 创建分支 |
| | `git checkout <branch>` | 切换分支 |
| | `git checkout -b <branch>` | 创建并切换分支 |
| | `git merge <branch>` | 合并分支 |
| | `git branch -d <branch>` | 删除分支 |
| **远程** | `git remote -v` | 查看远程仓库 |
| | `git fetch` | 拉取远程更新 |
| | `git pull` | 拉取并合并 |
| | `git push` | 推送到远程 |
| | `git push -u origin <branch>` | 推送并设置上游 |
| **历史** | `git log` | 查看历史 |
| | `git log --oneline` | 简洁历史 |
| | `git log --graph` | 图形化历史 |
| | `git show <commit>` | 查看提交详情 |
| **撤销** | `git checkout -- <file>` | 撤销工作区更改 |
| | `git reset HEAD <file>` | 取消暂存 |
| | `git reset --soft HEAD~1` | 撤销提交保留更改 |
| | `git reset --hard HEAD~1` | 撤销提交丢弃更改 |
| | `git revert <commit>` | 反转提交 |
| **暂存** | `git stash` | 暂存更改 |
| | `git stash list` | 查看暂存列表 |
| | `git stash pop` | 应用并删除暂存 |
| | `git stash apply` | 应用暂存 |
| **标签** | `git tag <name>` | 创建标签 |
| | `git tag -a <name> -m "msg"` | 创建附注标签 |
| | `git push --tags` | 推送标签 |
| **其他** | `git clean -fd` | 清理未跟踪文件 |
| | `git reflog` | 查看引用日志 |
| | `git grep "text"` | 搜索代码 |

---

## 1. Git 内部原理深入理解

### 1.1 Git 对象模型

Git 使用四种主要对象类型存储数据：

#### Blob 对象（文件内容）
```bash
# 查看文件对应的 blob 对象
git hash-object README.md

# 查看 blob 对象内容
git cat-file -p <blob-hash>

# 查看对象类型
git cat-file -t <object-hash>
```

#### Tree 对象（目录结构）
```bash
# 查看树对象
git cat-file -p master^{tree}

# 查看提交的树对象
git ls-tree -r HEAD
```

#### Commit 对象（提交信息）
```bash
# 查看提交对象详细信息
git cat-file -p HEAD

# 查看提交的父节点
git rev-parse HEAD^
git rev-parse HEAD~2
```

#### Tag 对象（标签）
```bash
# 创建附注标签（创建 tag 对象）
git tag -a v1.0.0 -m "Release version 1.0.0"

# 查看标签对象
git cat-file -p v1.0.0
```

### 1.2 引用与 reflog

**应用场景：** 恢复误删的分支或提交

```bash
# 查看所有引用
git show-ref

# 查看 HEAD 的历史记录
git reflog

# 恢复误删的提交
git reflog
git reset --hard HEAD@{2}  # 回到2步之前的状态

# 恢复误删的分支
git reflog show <branch-name>
git checkout -b <branch-name> HEAD@{n}
```

### 1.3 打包文件与垃圾回收

**应用场景：** 优化仓库大小，清理无用对象

```bash
# 查看对象数量
git count-objects -v

# 手动触发垃圾回收
git gc --aggressive --prune=now

# 查看打包文件
git verify-pack -v .git/objects/pack/*.idx | head -20

# 找出最大的文件
git verify-pack -v .git/objects/pack/*.idx | sort -k 3 -n | tail -10
```

---

## 2. 高级分支管理策略

### 2.1 Git Flow 工作流

**应用场景：** 大型团队的发布管理

```bash
# 初始化 git flow
git flow init

# 开始新功能开发
git flow feature start user-authentication

# 完成功能开发
git flow feature finish user-authentication

# 开始发布分支
git flow release start 1.2.0

# 完成发布
git flow release finish 1.2.0

# 紧急修复
git flow hotfix start critical-bug
git flow hotfix finish critical-bug
```

### 2.2 GitHub Flow 工作流

**应用场景：** 持续部署的敏捷团队

```bash
# 从 main 创建功能分支
git checkout -b feature/new-api main

# 定期推送并创建 PR
git push -u origin feature/new-api

# PR 合并后更新本地
git checkout main
git pull origin main
git branch -d feature/new-api
```

### 2.3 分支策略管理

#### 保护重要分支
```bash
# 通过 GitHub/GitLab 设置分支保护规则
# - 要求代码审查
# - 要求状态检查通过
# - 禁止强制推送
# - 要求线性历史

# 本地模拟分支保护
git config branch.main.pushRemote no_push
```

#### 分支清理自动化
```bash
# 删除已合并的本地分支
git branch --merged main | grep -v "main" | xargs git branch -d

# 删除远程已删除的本地追踪分支
git fetch --prune

# 批量删除远程分支
git branch -r --merged main | grep -v "main" | sed 's/origin\///' | xargs -I {} git push origin --delete {}
```

### 2.4 高级分支操作

**应用场景：** 复杂的分支重组

```bash
# 将分支 A 的提交移植到分支 B
git checkout feature-B
git cherry-pick <commit-from-A>

# 批量 cherry-pick
git cherry-pick <start-commit>..<end-commit>

# 重新设置分支基点
git checkout feature-branch
git rebase --onto main old-base feature-branch

# 示例：将 feature 从旧的 develop 分支移到新的 main 分支
git rebase --onto main develop feature
```

---

## 3. 交互式暂存与提交优化

### 3.1 交互式添加

**应用场景：** 精细控制每个文件的部分更改

```bash
# 进入交互式暂存模式
git add -i

# 交互式选择文件片段
git add -p <file>

# 在编辑器中精确选择行
git add -e <file>
```

**交互式暂存快捷键：**
- `y` - 暂存这个块
- `n` - 不暂存这个块
- `s` - 分割成更小的块
- `e` - 手动编辑块
- `q` - 退出

### 3.2 交互式 Rebase

**应用场景：** 清理和重组提交历史

```bash
# 交互式重写最近 5 个提交
git rebase -i HEAD~5

# 重新排序、合并、编辑提交
# 在编辑器中：
# pick   = 保留提交
# reword = 修改提交信息
# edit   = 修改提交内容
# squash = 合并到上一个提交
# fixup  = 合并到上一个提交但丢弃消息
# drop   = 删除提交
```

**实战示例：合并多个小提交**
```bash
# 假设有以下提交历史：
# abc123 Fix typo
# def456 Add tests
# ghi789 Implement feature
# jkl012 Fix linting

git rebase -i HEAD~4

# 编辑器中修改为：
# pick ghi789 Implement feature
# squash def456 Add tests
# squash abc123 Fix typo
# squash jkl012 Fix linting
```

### 3.3 提交信息规范

**应用场景：** 自动化生成 CHANGELOG

```bash
# 使用 Conventional Commits 规范
git commit -m "feat(auth): add OAuth2 login support"
git commit -m "fix(api): resolve null pointer exception"
git commit -m "docs(readme): update installation guide"
git commit -m "refactor(database): optimize query performance"
git commit -m "test(user): add unit tests for user service"
git commit -m "chore(deps): update dependencies"

# 使用 commitizen 工具
npm install -g commitizen cz-conventional-changelog
git cz  # 交互式生成规范的提交信息
```

---

## 4. 高级历史重写技巧

### 4.1 修改历史提交内容

**应用场景：** 从历史中移除敏感信息

```bash
# 修改特定提交
git rebase -i <commit>^
# 将对应行改为 'edit'，保存退出
git commit --amend
git rebase --continue

# 全局替换文件内容
git filter-branch --tree-filter 'rm -f passwords.txt' HEAD

# 使用 git-filter-repo（推荐，更快更安全）
pip install git-filter-repo
git filter-repo --path passwords.txt --invert-paths
```

### 4.2 修改提交作者信息

**应用场景：** 统一作者信息或修正错误配置

```bash
# 修改最后一次提交的作者
git commit --amend --author="New Author <email@example.com>"

# 批量修改历史作者信息
git filter-branch --env-filter '
if [ "$GIT_COMMITTER_EMAIL" = "old@example.com" ]; then
    export GIT_COMMITTER_NAME="New Name"
    export GIT_COMMITTER_EMAIL="new@example.com"
    export GIT_AUTHOR_NAME="New Name"
    export GIT_AUTHOR_EMAIL="new@example.com"
fi
' --tag-name-filter cat -- --all
```

### 4.3 拆分大提交

**应用场景：** 将一个大提交拆分为多个小提交

```bash
# 重置到要拆分的提交
git rebase -i <commit>^
# 标记为 'edit'

# 取消暂存所有文件
git reset HEAD^

# 分别添加和提交
git add file1.js
git commit -m "Part 1: Add feature A"

git add file2.js
git commit -m "Part 2: Add feature B"

# 继续 rebase
git rebase --continue
```

---

## 5. 复杂冲突解决方案

### 5.1 三方合并工具

**应用场景：** 可视化解决复杂冲突

```bash
# 配置合并工具（vimdiff/kdiff3/meld/p4merge）
git config --global merge.tool vimdiff
git config --global mergetool.prompt false

# 使用合并工具
git mergetool

# VS Code 作为合并工具
git config --global merge.tool vscode
git config --global mergetool.vscode.cmd 'code --wait $MERGED'
```

### 5.2 智能冲突解决策略

**应用场景：** 批量处理冲突

```bash
# 使用 ours 策略（保留当前分支）
git merge -X ours feature-branch

# 使用 theirs 策略（采用合并分支）
git merge -X theirs feature-branch

# 仅针对特定文件使用策略
git checkout --ours path/to/file
git checkout --theirs path/to/file
git add path/to/file
```

### 5.3 Rerere（重用记录的冲突解决）

**应用场景：** 自动应用之前的冲突解决方案

```bash
# 启用 rerere
git config --global rerere.enabled true

# Git 会自动记录和重用冲突解决方案
# 查看 rerere 缓存
git rerere status
git rerere diff

# 清除 rerere 缓存
git rerere forget <pathspec>
```

---

## 6. 高级搜索与调试

### 6.1 Git Grep 高级搜索

**应用场景：** 在整个仓库历史中搜索代码

```bash
# 在所有文件中搜索
git grep "function getUserData"

# 在特定提交中搜索
git grep "TODO" <commit-hash>

# 显示行号
git grep -n "import React"

# 搜索整个单词
git grep -w "user"

# 统计匹配数量
git grep -c "console.log"

# 在特定类型文件中搜索
git grep "error" -- "*.js"

# 正则表达式搜索
git grep -E "class \w+Controller"
```

### 6.2 Git Bisect 二分查找 Bug

**应用场景：** 快速定位引入 bug 的提交

```bash
# 开始二分查找
git bisect start

# 标记当前版本为坏版本
git bisect bad

# 标记已知的好版本
git bisect good v1.2.0

# Git 会自动切换到中间提交，测试后标记
git bisect good  # 或 git bisect bad

# 自动化测试
git bisect run npm test

# 结束 bisect
git bisect reset
```

**实战示例：**
```bash
# 编写自动化测试脚本
cat > test.sh << 'EOF'
#!/bin/bash
npm test
if [ $? -eq 0 ]; then
    exit 0  # 测试通过
else
    exit 1  # 测试失败
fi
EOF

chmod +x test.sh
git bisect start HEAD v1.0.0
git bisect run ./test.sh
```

### 6.3 Git Blame 深度分析

**应用场景：** 追踪代码变更历史

```bash
# 查看文件每行的最后修改者
git blame <file>

# 显示邮箱和时间
git blame -e -t <file>

# 查看特定行范围
git blame -L 10,20 <file>

# 忽略空白更改
git blame -w <file>

# 查看删除的代码
git log -S "deleted_function" --source --all

# 追踪函数的演变
git log -L :functionName:file.js
```

---

## 7. 子模块与 Subtree 管理

### 7.1 Git Submodule

**应用场景：** 引用外部依赖仓库

```bash
# 添加子模块
git submodule add https://github.com/user/repo.git libs/repo

# 克隆包含子模块的仓库
git clone --recurse-submodules <repo-url>

# 初始化并更新子模块
git submodule init
git submodule update

# 更新所有子模块到最新版本
git submodule update --remote

# 在子模块中工作
cd libs/repo
git checkout main
git pull
cd ../..
git add libs/repo
git commit -m "Update submodule"

# 删除子模块
git submodule deinit libs/repo
git rm libs/repo
rm -rf .git/modules/libs/repo
```

### 7.2 Git Subtree

**应用场景：** 将外部项目嵌入主仓库

```bash
# 添加远程仓库
git remote add vendor-lib https://github.com/vendor/lib.git

# 添加 subtree
git subtree add --prefix=vendor/lib vendor-lib main --squash

# 更新 subtree
git subtree pull --prefix=vendor/lib vendor-lib main --squash

# 推送更改回上游
git subtree push --prefix=vendor/lib vendor-lib main
```

**Submodule vs Subtree 对比：**

| 特性 | Submodule | Subtree |
|------|-----------|---------|
| 历史记录 | 分离 | 合并 |
| 克隆复杂度 | 需要额外命令 | 简单 |
| 更新操作 | 复杂 | 相对简单 |
| 仓库大小 | 小 | 大 |
| 适用场景 | 清晰的依赖边界 | 需要修改外部代码 |

---

## 8. 性能优化与仓库维护

### 8.1 仓库体积优化

**应用场景：** 清理臃肿的仓库

```bash
# 查看仓库大小
git count-objects -vH

# 查找大文件
git rev-list --objects --all | \
  git cat-file --batch-check='%(objecttype) %(objectname) %(objectsize) %(rest)' | \
  sed -n 's/^blob //p' | \
  sort --numeric-sort --key=2 | \
  tail -n 10

# 使用 BFG Repo-Cleaner 删除大文件
java -jar bfg.jar --strip-blobs-bigger-than 50M <repo>

# 或使用 git-filter-repo
git filter-repo --strip-blobs-bigger-than 10M

# 清理后的垃圾回收
git reflog expire --expire=now --all
git gc --prune=now --aggressive
```

### 8.2 Shallow Clone（浅克隆）

**应用场景：** 加速 CI/CD 构建

```bash
# 只克隆最近 1 次提交
git clone --depth 1 <repo-url>

# 取消浅克隆限制
git fetch --unshallow

# 只克隆特定分支
git clone --single-branch --branch main <repo-url>
```

### 8.3 Sparse Checkout（稀疏检出）

**应用场景：** 只检出大仓库的部分文件

```bash
# 初始化稀疏检出
git clone --no-checkout <repo-url>
cd <repo>
git sparse-checkout init --cone

# 指定要检出的目录
git sparse-checkout set src/components src/utils

# 查看当前配置
git sparse-checkout list

# 添加更多目录
git sparse-checkout add docs
```

---

## 9. 团队协作高级实践

### 9.1 代码审查最佳实践

**应用场景：** 提高代码质量

```bash
# 创建审查分支
git checkout -b review/feature-x origin/feature-x

# 查看更改摘要
git diff main...feature-x --stat

# 逐文件审查
git diff main...feature-x -- path/to/file

# 审查特定提交
git show <commit-hash>

# 添加审查注释（使用 GitHub/GitLab PR 评论）
# 批准并合并
git checkout main
git merge --no-ff feature-x
```

### 9.2 多人协作冲突预防

**应用场景：** 减少合并冲突

```bash
# 在推送前先拉取并 rebase
git pull --rebase origin main

# 配置默认使用 rebase
git config --global pull.rebase true

# 使用 pre-push hook 强制检查
cat > .git/hooks/pre-push << 'EOF'
#!/bin/bash
current_branch=$(git symbolic-ref HEAD | sed -e 's,.*/\(.*\),\1,')
git fetch origin $current_branch
if ! git diff origin/$current_branch --quiet; then
    echo "Remote has changes. Please pull first."
    exit 1
fi
EOF
chmod +x .git/hooks/pre-push
```

### 9.3 发布管理

**应用场景：** 自动化版本发布

```bash
# 创建发布标签
git tag -a v1.2.0 -m "Release version 1.2.0"
git push origin v1.2.0

# 生成 CHANGELOG
git log v1.1.0..v1.2.0 --oneline > CHANGELOG.md

# 使用 conventional-changelog 自动生成
npm install -g conventional-changelog-cli
conventional-changelog -p angular -i CHANGELOG.md -s

# 创建 GitHub Release
gh release create v1.2.0 --generate-notes
```

---

## 10. Git Hooks 自动化

### 10.1 客户端 Hooks

**pre-commit - 提交前代码检查**
```bash
cat > .git/hooks/pre-commit << 'EOF'
#!/bin/bash

# 运行代码格式化
npm run format

# 运行 Linter
npm run lint
if [ $? -ne 0 ]; then
    echo "Linting failed. Please fix errors before committing."
    exit 1
fi

# 运行测试
npm test
if [ $? -ne 0 ]; then
    echo "Tests failed. Please fix before committing."
    exit 1
fi
EOF

chmod +x .git/hooks/pre-commit
```

**commit-msg - 提交信息验证**
```bash
cat > .git/hooks/commit-msg << 'EOF'
#!/bin/bash

commit_msg=$(cat "$1")

# 验证 Conventional Commits 格式
if ! echo "$commit_msg" | grep -qE "^(feat|fix|docs|style|refactor|test|chore)(\(.+\))?: .+"; then
    echo "Error: Commit message must follow Conventional Commits format"
    echo "Example: feat(auth): add OAuth2 support"
    exit 1
fi
EOF

chmod +x .git/hooks/commit-msg
```

**pre-push - 推送前检查**
```bash
cat > .git/hooks/pre-push << 'EOF'
#!/bin/bash

# 禁止直接推送到 main
protected_branch='main'
current_branch=$(git symbolic-ref HEAD | sed -e 's,.*/\(.*\),\1,')

if [ "$current_branch" = "$protected_branch" ]; then
    echo "Direct push to $protected_branch is not allowed!"
    exit 1
fi

# 运行完整测试套件
npm run test:all
if [ $? -ne 0 ]; then
    echo "Tests failed. Push aborted."
    exit 1
fi
EOF

chmod +x .git/hooks/pre-push
```

### 10.2 服务器端 Hooks

**pre-receive - 服务器端接收前检查**
```bash
cat > hooks/pre-receive << 'EOF'
#!/bin/bash

while read oldrev newrev refname; do
    # 禁止删除 main 分支
    if [ "$refname" = "refs/heads/main" ] && [ "$newrev" = "0000000000000000000000000000000000000000" ]; then
        echo "Error: Deleting main branch is not allowed"
        exit 1
    fi

    # 禁止强制推送
    if [ "$oldrev" != "0000000000000000000000000000000000000000" ]; then
        if ! git merge-base --is-ancestor "$oldrev" "$newrev"; then
            echo "Error: Force push is not allowed"
            exit 1
        fi
    fi
done
EOF

chmod +x hooks/pre-receive
```

### 10.3 使用 Husky 管理 Hooks

**应用场景：** 团队共享 Git Hooks

```bash
# 安装 Husky
npm install --save-dev husky

# 初始化 Husky
npx husky init

# 添加 pre-commit hook
npx husky add .husky/pre-commit "npm test"

# 添加 commit-msg hook（配合 commitlint）
npm install --save-dev @commitlint/{config-conventional,cli}
npx husky add .husky/commit-msg 'npx --no -- commitlint --edit "$1"'
```

---

## 11. 实战场景案例

### 11.1 场景：紧急修复生产 Bug

**问题：** 生产环境发现严重 bug，需要立即修复并发布

```bash
# 1. 从生产标签创建 hotfix 分支
git checkout -b hotfix/critical-bug v1.2.0

# 2. 修复 bug 并提交
git add .
git commit -m "fix: resolve critical null pointer exception"

# 3. 创建新版本标签
git tag -a v1.2.1 -m "Hotfix release"

# 4. 合并回 main 和 develop
git checkout main
git merge --no-ff hotfix/critical-bug
git push origin main --tags

git checkout develop
git merge --no-ff hotfix/critical-bug
git push origin develop

# 5. 删除 hotfix 分支
git branch -d hotfix/critical-bug
```

### 11.2 场景：恢复误删的代码

**问题：** 不小心删除了重要代码，已经提交但未推送

```bash
# 1. 查看 reflog 找到删除前的提交
git reflog

# 2. 恢复文件
git checkout <commit-hash> -- path/to/file

# 3. 如果已经推送，创建恢复提交
git revert <bad-commit-hash>
git push origin main
```

### 11.3 场景：重写提交历史移除敏感信息

**问题：** 不小心提交了包含密码的配置文件

```bash
# 1. 使用 git-filter-repo（推荐）
pip install git-filter-repo
git filter-repo --path config/secrets.yml --invert-paths

# 2. 强制推送（警告：会重写历史）
git push origin --force --all
git push origin --force --tags

# 3. 通知团队成员重新克隆仓库
# 或者使用 git pull --rebase 更新
```

### 11.4 场景：合并长期存在的功能分支

**问题：** 功能分支开发时间长，与 main 分支差异大

```bash
# 1. 确保功能分支是最新的
git checkout feature/long-running
git fetch origin
git rebase origin/main

# 2. 解决所有冲突
git mergetool
git rebase --continue

# 3. 运行完整测试
npm run test:all

# 4. 使用 squash merge 保持历史清晰
git checkout main
git merge --squash feature/long-running
git commit -m "feat: implement complete user management system"

# 5. 推送并删除功能分支
git push origin main
git branch -d feature/long-running
git push origin --delete feature/long-running
```

### 11.5 场景：迁移到 Monorepo

**问题：** 将多个独立仓库合并为一个 monorepo

```bash
# 1. 创建 monorepo 结构
mkdir monorepo && cd monorepo
git init

# 2. 添加第一个项目
git remote add project-a https://github.com/user/project-a.git
git fetch project-a
git merge -s ours --no-commit --allow-unrelated-histories project-a/main
git read-tree --prefix=packages/project-a/ -u project-a/main
git commit -m "Merge project-a into monorepo"

# 3. 添加第二个项目
git remote add project-b https://github.com/user/project-b.git
git fetch project-b
git merge -s ours --no-commit --allow-unrelated-histories project-b/main
git read-tree --prefix=packages/project-b/ -u project-b/main
git commit -m "Merge project-b into monorepo"

# 4. 清理远程引用
git remote remove project-a
git remote remove project-b
```

### 11.6 场景：实现 CI/CD 中的增量构建

**问题：** 只构建发生变化的包

```bash
# 1. 检测变化的文件
changed_packages=$(git diff --name-only HEAD~1 HEAD | grep "^packages/" | cut -d'/' -f2 | sort -u)

# 2. 只测试变化的包
for pkg in $changed_packages; do
    cd packages/$pkg
    npm test
    cd ../..
done

# 3. 使用 Lerna 或 Nx 进行智能构建
npx lerna run test --since HEAD~1
npx nx affected:test --base=HEAD~1
```

### 11.7 场景：管理多环境配置

**问题：** 针对不同环境维护不同配置

```bash
# 1. 使用分支策略
git checkout -b config/production
# 修改生产配置
git commit -am "chore: production configuration"

git checkout -b config/staging
# 修改预发布配置
git commit -am "chore: staging configuration"

# 2. 部署时合并配置
git checkout main
git merge --no-commit --no-ff config/production
# 部署到生产环境

# 3. 使用 git-crypt 加密敏感配置
git-crypt init
echo "*.secret.yml filter=git-crypt diff=git-crypt" >> .gitattributes
git add .gitattributes
git commit -m "chore: enable git-crypt for secrets"
```

---

## 总结

Git 的高阶功能远不止基本的 add、commit、push。掌握这些高级技巧可以：

1. **提高效率** - 通过别名、hooks、自动化脚本减少重复工作
2. **保证质量** - 通过 bisect、blame、审查流程提升代码质量
3. **团队协作** - 通过规范的工作流、分支策略提高协作效率
4. **问题恢复** - 通过 reflog、revert、reset 快速恢复错误
5. **性能优化** - 通过仓库清理、浅克隆、稀疏检出提升性能

**最佳实践建议：**

- ✅ 始终使用有意义的提交信息
- ✅ 定期清理本地和远程分支
- ✅ 在重写历史前备份
- ✅ 使用 `.gitignore` 避免提交无关文件
- ✅ 利用 Git Hooks 自动化检查
- ✅ 学习使用 `git reflog` 恢复误操作
- ⚠️ 慎用 `--force` 推送
- ⚠️ 不要重写已推送的公共历史

记住：Git 是一个强大的工具，但强大的功能也意味着需要谨慎使用。在团队环境中，始终遵循约定的工作流程和最佳实践。
