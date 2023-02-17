/*
 * This Java source file was generated by the Gradle 'init' task.
 */
package deploy;

import org.pytorch.IValue;
import org.pytorch.Module;
import org.pytorch.Tensor;

import java.util.Arrays;

public class App {
    public boolean someAppMethod() {
        return true;
    }

  public static void main(String[] args) {

    // Load PyTorch Geometric Dependencies
    System.loadLibrary("torchsparse"); //NOTE: This must be done in a static context, e.g., in main.
    System.loadLibrary("torchscatter");
    System.loadLibrary("torchcluster");
    System.loadLibrary("torchsplineconv");

    // Check PyTorch Dependency
    Tensor data =
        Tensor.fromBlob(
            new int[] {1, 2, 3, 4, 5, 6}, // data
            new long[] {2, 3} // shape
            );
    System.out.println("DEBUGGING: data pytorch tensor = "+Arrays.toString(data.getDataAsIntArray()));

    // Check args
    if (args.length<=0) {System.out.println("Usage: Library </path/to/pytorch/model.pt>"); System.exit(0); }
    String path = args[0];
    
    // Load and apply model
    Module mod = Module.load(path);
    Tensor x =
        Tensor.fromBlob(
            new float[] {1, 2, 3, 4, 5, 6, 7, 1, 2, 3, 4, 5, 6, 7, 1, 2, 3, 4, 5, 6, 7}, // data //NOTE: x must have type float
            new long[] {3, 7} // shape
            );
    Tensor edge_index =
        Tensor.fromBlob(
            new long[] {0, 1, 2, 1, 2, 0}, // data //NOTE: edge_index must have type long
            new long[] {2, 3} // shape
            );
    IValue result = mod.forward(IValue.from(x), IValue.from(edge_index));
    Tensor output = result.toTensor();
    System.out.println("shape: " + Arrays.toString(output.shape()));
    System.out.println("data: " + Arrays.toString(output.getDataAsFloatArray()));

    // Workaround for https://github.com/facebookincubator/fbjni/issues/25
    System.exit(0);

  } // public static void main()

} // public class App
