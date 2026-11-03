import { HealthStatus, LoanApplicationData, ModelInfo, PredictionResponse } from "@/types/loan";

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export class ApiError extends Error {
  status?: number;
  constructor(message: string, status?: number) {
    super(message);
    this.name = "ApiError";
    this.status = status;
  }
}

export async function fetchHealth(): Promise<HealthStatus> {
  try {
    const res = await fetch(`${API_BASE_URL}/health`, {
      method: "GET",
      cache: "no-store",
    });
    if (!res.ok) {
      throw new ApiError(`Health check failed with status ${res.status}`, res.status);
    }
    return await res.json();
  } catch (err: any) {
    if (err instanceof ApiError) throw err;
    throw new ApiError("Unable to reach backend service. Ensure FastAPI server is running on port 8000.");
  }
}

export async function fetchModelInfo(): Promise<ModelInfo> {
  try {
    const res = await fetch(`${API_BASE_URL}/model-info`, {
      method: "GET",
      cache: "no-store",
    });
    if (!res.ok) {
      throw new ApiError(`Model info failed with status ${res.status}`, res.status);
    }
    return await res.json();
  } catch (err: any) {
    if (err instanceof ApiError) throw err;
    throw new ApiError("Failed to fetch model metadata from API.");
  }
}

export async function submitLoanPrediction(data: LoanApplicationData): Promise<PredictionResponse> {
  try {
    const res = await fetch(`${API_BASE_URL}/predict`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(data),
    });

    const responseData = await res.json();

    if (!res.ok) {
      const errorMsg =
        responseData?.detail && typeof responseData.detail === "string"
          ? responseData.detail
          : "Prediction request failed. Please check form inputs.";
      throw new ApiError(errorMsg, res.status);
    }

    return responseData as PredictionResponse;
  } catch (err: any) {
    if (err instanceof ApiError) throw err;
    throw new ApiError(
      "Network connection error: Unable to connect to the prediction backend at " +
        API_BASE_URL +
        ". Verify that the backend is running."
    );
  }
}

