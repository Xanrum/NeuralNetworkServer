public class LoadInputsRequest
{
    public List<LoadInputsRequestItem> Data { get; set; }
}

public class LoadInputsRequestItem {
    public string Key { get; set; }
    public double[] Inputs { get; set; }
}