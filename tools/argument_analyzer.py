import ast
from typing import List, Dict, Set, Union

class ArgumentUsageAnalyzer:
    def __init__(self, args_namespace_object):
        self.args_obj = args_namespace_object
        self.defined_args = set(vars(self.args_obj).keys())
        self.code_used_args = set()
        self.metadata_args_spec = set()

    def _get_str_from_ast_node(self, node: ast.AST) -> Union[str, None]:
        """ASTノードから文字列リテラルを抽出するヘルパー"""
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return node.value
        elif isinstance(node, ast.Str): # Python < 3.8
            return node.s
        return None

    def analyze_file(self, file_path: str) -> None:
        """ファイル内のargsオブジェクトへのアクセスを解析"""
        with open(file_path, 'r', encoding='utf-8') as f:
            try:
                tree = ast.parse(f.read(), filename=file_path)
            except SyntaxError as e:
                print(f"SyntaxError in {file_path}: {e}")
                return

        # TODO: argsオブジェクトのエイリアスを追跡する基本的なメカニズム
        # 簡単な例: current_arg_names = {'args', 'cfg', ...}
        # 代入文 (new_name = old_name) を見つけて更新する
        arg_obj_names = {'args'} # 解析対象とするargsオブジェクトの名前のセット

        for node in ast.walk(tree):
            # 直接アクセス (obj.attr)
            if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id in arg_obj_names:
                self.code_used_args.add(node.attr)

            # 動的アクセス (getattr(obj, "attr_str" [, default]))
            elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == 'getattr':
                if len(node.args) >= 2 and isinstance(node.args[0], ast.Name) and node.args[0].id in arg_obj_names:
                    attr_name = self._get_str_from_ast_node(node.args[1])
                    if attr_name:
                        self.code_used_args.add(attr_name)

            # 辞書アクセス (vars(obj)["attr_str"]) - より複雑なパターンマッチングが必要
            # 現状は vars(args) の呼び出しのみを大まかに捉える
            # TODO: vars(args) の結果を使った具体的なキーアクセスを追跡する

            # hasattr(obj, "attr_str")
            elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == 'hasattr':
                if len(node.args) == 2 and isinstance(node.args[0], ast.Name) and node.args[0].id in arg_obj_names:
                    attr_name = self._get_str_from_ast_node(node.args[1])
                    if attr_name:
                        self.code_used_args.add(attr_name) # hasattrでチェックされる属性も使用とみなす

            # vars(obj) - 大雑把な扱い
            elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == 'vars':
                if node.args and isinstance(node.args[0], ast.Name) and node.args[0].id in arg_obj_names:
                    # vars()が使われた場合、現状維持で全定義済み引数を追加
                    # または、この事実を記録し、レポートで注意を促す
                    self.code_used_args.update(self.defined_args)
                    # print(f"INFO: vars({node.args[0].id}) used in {file_path}. Assuming all defined args might be used.")

    def set_metadata_args_specification(self, metadata_args_list: List[str]) -> None:
        """メタデータで使用されると指定された引数のリストを設定"""
        self.metadata_args_spec = set(metadata_args_list)

    def get_analysis_results(self) -> Dict[str, Set[str]]:
        """解析結果を取得"""
        # code_used_args が defined_args に本当に存在するかを再確認するステップを挟むのもあり
        # (AST解析が完璧でない場合、存在しない属性名を拾う可能性を考慮)
        # validated_code_used_args = {arg for arg in self.code_used_args if hasattr(self.args_obj, arg)}
        # ただし、これは実行時の情報であり、静的解析の趣旨とは少し異なる

        results = {
            'undefined_accessed': self.code_used_args - self.defined_args,
            'unused_defined': self.defined_args - self.code_used_args,
            'metadata_spec_not_defined': self.metadata_args_spec - self.defined_args,
            'metadata_spec_unused_in_code': self.metadata_args_spec - self.code_used_args,
            'code_used_not_in_metadata_spec': self.code_used_args - self.metadata_args_spec
        }
        return results

    def print_analysis(self) -> None:
        """解析結果を表示"""
        results = self.get_analysis_results()

        print("\n=== 引数の使用状況分析 ===")

        if results['undefined_accessed']:
            print("\n警告: コード内でアクセスされているが、(実行時のargsオブジェクトには)未定義の引数:")
            for arg in sorted(list(results['undefined_accessed'])): print(f"  - {arg}")

        if results['unused_defined']:
            print("\n情報: (実行時のargsオブジェクトで)定義されているが、コード内での使用が確認できなかった引数:")
            for arg in sorted(list(results['unused_defined'])): print(f"  - {arg}")

        if results['metadata_spec_not_defined']:
            print("\n警告: メタデータ仕様に含まれるが、(実行時のargsオブジェクトには)未定義の引数:")
            for arg in sorted(list(results['metadata_spec_not_defined'])): print(f"  - {arg}")

        if results['metadata_spec_unused_in_code']:
            print("\n情報: メタデータ仕様に含まれるが、コード内での使用が確認できなかった引数:")
            for arg in sorted(list(results['metadata_spec_unused_in_code'])): print(f"  - {arg}")

        if results['code_used_not_in_metadata_spec']:
            print("\n情報: コード内で使用されているが、メタデータ仕様に含まれていない引数:")
            for arg in sorted(list(results['code_used_not_in_metadata_spec'])): print(f"  - {arg}")

        print("\n=== 統計 ===")
        print(f"実行時argsの定義済み引数: {len(self.defined_args)}")
        print(f"コード内で使用が確認された引数 (ASTベース): {len(self.code_used_args)}")
        print(f"メタデータ仕様の引数: {len(self.metadata_args_spec)}")
        print("=====================\n")