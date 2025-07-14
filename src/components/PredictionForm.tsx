import React, { useState } from 'react';
import { Button } from '@/components/ui/button'; // Assuming you have these
import { Input } from '@/components/ui/input';   // Assuming you have these
import { Label } from '@/components/ui/label';   // Assuming you have these

// These are the features our model expects, from backend/app.py
const FEATURE_COLUMNS = [
    'age', 'player_height', 'player_weight', 'gp', 'pts', 'reb', 'ast',
    'net_rating', 'oreb_pct', 'dreb_pct', 'usg_pct', 'ts_pct', 'ast_pct'
];

interface FormData {
    [key: string]: string;
}

const initialFormData: FormData = FEATURE_COLUMNS.reduce((obj, col) => {
    obj[col] = '';
    return obj;
}, {} as FormData);

const PredictionForm: React.FC = () => {
    const [formData, setFormData] = useState<FormData>(initialFormData);
    const [prediction, setPrediction] = useState<number | null>(null);
    const [error, setError] = useState<string | null>(null);
    const [isLoading, setIsLoading] = useState<boolean>(false);

    const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
        setFormData({
            ...formData,
            [e.target.name]: e.target.value,
        });
    };

    const handleSubmit = async (e: React.FormEvent<HTMLFormElement>) => {
        e.preventDefault();
        setIsLoading(true);
        setPrediction(null);
        setError(null);

        // Convert form data strings to numbers
        const numericFormData: { [key: string]: number } = {};
        let conversionError = false;
        for (const key in formData) {
            const value = parseFloat(formData[key]);
            if (isNaN(value)) {
                setError(`Invalid input for ${key}. Please enter a valid number.`);
                conversionError = true;
                break;
            }
            numericFormData[key] = value;
        }

        if (conversionError) {
            setIsLoading(false);
            return;
        }

        try {
            const response = await fetch('/api/predict', { // Proxied request
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(numericFormData),
            });

            const result = await response.json();

            if (response.ok && result.status === 'success') {
                setPrediction(result.predicted_probability_hall_of_fame);
            } else {
                setError(result.message || 'An error occurred during prediction.');
            }
        } catch (err) {
            console.error("Prediction API call failed:", err);
            setError('Failed to connect to the prediction service. Please try again later.');
        } finally {
            setIsLoading(false);
        }
    };

    return (
        <div className="max-w-md mx-auto p-4 space-y-6">
            <h2 className="text-2xl font-semibold text-center">Hall of Fame Predictor</h2>
            <form onSubmit={handleSubmit} className="space-y-4">
                {FEATURE_COLUMNS.map((col) => (
                    <div key={col} className="space-y-1">
                        <Label htmlFor={col} className="capitalize">{col.replace('_', ' ')}</Label>
                        <Input
                            type="number" // Using number type for stats
                            step="any"     // Allow decimals
                            id={col}
                            name={col}
                            value={formData[col]}
                            onChange={handleChange}
                            placeholder={`Enter ${col.replace('_', ' ')}`}
                            required
                            className="w-full"
                        />
                    </div>
                ))}
                <Button type="submit" disabled={isLoading} className="w-full">
                    {isLoading ? 'Predicting...' : 'Predict HOF Probability'}
                </Button>
            </form>
            {prediction !== null && (
                <div className="mt-6 p-4 bg-green-100 border border-green-400 text-green-700 rounded">
                    <h3 className="font-semibold">Prediction Result:</h3>
                    <p>Predicted Hall of Fame Probability: <span className="font-bold">{(prediction * 100).toFixed(2)}%</span></p>
                </div>
            )}
            {error && (
                <div className="mt-6 p-4 bg-red-100 border border-red-400 text-red-700 rounded">
                    <h3 className="font-semibold">Error:</h3>
                    <p>{error}</p>
                </div>
            )}
        </div>
    );
};

export default PredictionForm;
