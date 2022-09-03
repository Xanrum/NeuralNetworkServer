using ILGPU;
using ILGPU.Runtime;
using ILGPU.Runtime.CPU;
using ILGPU.Runtime.Cuda;
using System.Numerics;
using System.Runtime.Intrinsics;
using System.Threading.Channels;

namespace NeuralNetworkServer;

public class NeuralNetworkCalcerCuda
{
    public NeuralNetworkCalcerCuda()
    {
        for (int i = 0; i < 1; i++)
        {
            var worker = new Worker();
            Context context = Context.CreateDefault();
            worker.accelerator = context.CreateCudaAccelerator(0);
            Console.WriteLine(worker.accelerator.Name);
        
            worker.Kernel = worker.accelerator.LoadAutoGroupedStreamKernel(
            (Index1D jobIndex, ArrayView<float> inputs, ArrayView<int> indexes, ArrayView<int> model, ArrayView<float> synapses, ArrayView<float> output) => {
                var firstLevel = model[0];
                var startIndex = indexes[jobIndex];
                var prev = new float[180];
                var next = new float[180];
                for (var i = 0; i < firstLevel; i++) prev[i] = inputs[startIndex + i];
                var synapseIndex = 0;
                var modelLen = model.Length;
                for (var i = 1; i < modelLen; i++)
                {
                    var prevCount = model[i - 1];
                    var currentLayerLen = model[i];
                    for (var j = 0; j < currentLayerLen; j++)
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
                var lastLevel = model[modelLen-1];
                for (var i = 0; i < lastLevel; i++)
                {
                    output[jobIndex * lastLevel + i] = prev[i];
                }
            });
            workers.Writer.TryWrite(worker);
        }
    }

    private Channel<Worker> workers = Channel.CreateUnbounded<Worker>();

    class Worker
    {
        public Accelerator accelerator;
        public Action<Index1D, ArrayView<float>, ArrayView<int>, ArrayView<int>, ArrayView<float>, ArrayView<float>> Kernel;
    }


    public async Task<float[][]> Calc(float[] inputs, int[] indexes, float[] synapses, int[] model)
    {
        var worker = await workers.Reader.ReadAsync();
        try
        {
            var accelerator = worker.accelerator;
            var outCount = model.Last();
            var deviceInputs = accelerator.Allocate1D(inputs);
            var deviceIndexes = accelerator.Allocate1D(indexes);
            var deviceSynapses = accelerator.Allocate1D(synapses);
            var deviceModel = accelerator.Allocate1D(model);
            var deviceOutput = accelerator.Allocate1D<float>(outCount * indexes.Length);
            
            worker.Kernel(indexes.Length, deviceInputs.View, deviceIndexes.View, deviceModel.View, deviceSynapses.View, deviceOutput.View);
                accelerator.Synchronize();
            var res = deviceOutput.GetAsArray1D();

            // accelerator.Dispose();
            // context.Dispose();
            var result = new float[indexes.Length][];
            for (int i = 0; i < indexes.Length; i++)
            {
                var r = result[i] = new float[outCount];
                Array.Copy(res, i * outCount, r, 0, outCount);
            }
            return result;
        }
        finally
        {
            workers.Writer.TryWrite(worker);
        }
    }
}