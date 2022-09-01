using ILGPU;
using ILGPU.Runtime;
using ILGPU.Runtime.CPU;
using ILGPU.Runtime.Cuda;
using System.Numerics;
using System.Runtime.Intrinsics;

namespace NeuralNetworkServer;

public class NeuralNetworkCalcerCuda
{
    public NeuralNetworkCalcerCuda()
    {
        Context context = Context.CreateDefault();
        accelerator = context.CreateCudaAccelerator(0);
        Console.WriteLine(accelerator.Name);
        
        Kernel = accelerator.LoadAutoGroupedStreamKernel(
        (Index1D jobIndex, ArrayView<float> inputs, ArrayView<int> indexes, ArrayView<int> model, ArrayView<float> synapses, ArrayView<float> output) => {
            var firstLevel = model[0];
            var startIndex = indexes[jobIndex];
            var prev = new float[300];
            var next = new float[300];
            for (int i = 0; i < firstLevel; i++) prev[i] = inputs[startIndex + i];
            var synapseIndex = 0;
            for (var i = 1; i < model.Length; i++)
            {
                var prevCount = model[i - 1];
                for (var j = 0; j < model[i]; j++)
                {
                    var sum = 0f;
                    for (int k = 0; k < prevCount; k++)
                    {
                        sum += prev[k] * synapses[synapseIndex];
                        synapseIndex++;
                    }
                    next[j] = MathF.Tanh(sum);
                }
                (next, prev) = (prev, next);
            }
            var lastLevel = model[model.Length-1];
            for (int i = 0; i < lastLevel; i++)
            {
                output[jobIndex * lastLevel + i] = prev[i];
            }
        });
    }

    private Accelerator accelerator;
    private Action<Index1D, ArrayView<float>, ArrayView<int>, ArrayView<int>, ArrayView<float>, ArrayView<float>> Kernel;


    public float[][] Calc(float[] inputs, int[] indexes, float[] synapses, int[] model)
    {
        var outCount = model.Last();
        var deviceInputs = accelerator.Allocate1D(inputs);
        var deviceIndexes = accelerator.Allocate1D(indexes);
        var deviceSynapses = accelerator.Allocate1D(synapses);
        var deviceModel = accelerator.Allocate1D(model);
        var deviceOutput = accelerator.Allocate1D<float>(outCount * indexes.Length);
        
        // load / compile the kernel
     
        lock (accelerator)
        {
            Kernel(indexes.Length, deviceInputs.View, deviceIndexes.View, deviceModel.View, deviceSynapses.View, deviceOutput.View);
            accelerator.Synchronize();
        }
        var res = deviceOutput.GetAsArray1D();

        // accelerator.Dispose();
        // context.Dispose();
        var result = new float[indexes.Length][];
        for (int i = 0; i < indexes.Length; i++)
        {
            var r = result[i] = new float[outCount];
            Array.Copy(res, i*outCount, r, 0, outCount);
        }
        return result;
    }
}