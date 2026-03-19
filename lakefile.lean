import Lake
open Lake DSL

package «mnist» where
  version := v!"0.1.0"
  buildType := .release

lean_exe «mnist-mlp» where
  root := `Main_working_1d_s4tf

lean_exe «mnist-cnn» where
  root := `Main_working_2d_s4tf
