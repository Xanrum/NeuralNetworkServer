using Microsoft.AspNetCore.Mvc;
using System.Collections.Concurrent;

namespace NeuralNetworkServer.Controllers;

[ApiController]
[Route("[controller]")]
public class NeuralController : ControllerBase
{

    private static readonly ConcurrentDictionary<long, double[]> InputsCache = new();

    [HttpPost("Load")]
    public void Load(LoadInputsRequest request)
    {
        foreach (var item in request.Data)
        {
            InputsCache[item.Key] = item.Inputs;
        }
    }
    
    [HttpPost("Calc")]
    public async Task<CalcResponse> Calc()
    {
        Request.EnableBuffering();
        Request.Body.Position = 0;
        var ms = new MemoryStream();
        await Request.Body.CopyToAsync(ms);
        ms.Position = 0;
        var binaryReader = new BinaryReader(ms);
        var modelLength = binaryReader.ReadInt32();
        var model = new int[modelLength];
        for (var i = 0; i < modelLength; i++)
        {
            model[i] = binaryReader.ReadInt32();
        }
        var inputsLength = binaryReader.ReadInt32();
        var requestInputs = new long[inputsLength];
        for (var i = 0; i < inputsLength; i++)
        {
            requestInputs[i] = binaryReader.ReadInt64();
        }
        
        var synapseLength = binaryReader.ReadInt32();
        var synapses = new double[synapseLength];
        for (var i = 0; i < synapseLength; i++)
        {
            synapses[i] = binaryReader.ReadDouble();
        }
        var inputs = new List<double[]>();
        var unknownInputs = new List<long>();
        foreach (var s in requestInputs)
        {
            if (InputsCache.TryGetValue(s, out var input))
            {
                inputs.Add(input);
            }
            else
            {
                unknownInputs.Add(s);
            }
        }
        if (unknownInputs.Count != 0)
            return new()
            {
                UnkownInputs = unknownInputs,
                Outputs = Array.Empty<double[]>()
            };
        var calcer = new NeuralNetworkCalcer(model);
        var outputs = new double[inputs.Count][];
        for (var i = 0; i < inputs.Count; i++)
        {
            outputs[i] = calcer.Calc(inputs[i], synapses);
        }
        return new()
        {
            UnkownInputs = unknownInputs,
            Outputs = outputs
        };
    }
}