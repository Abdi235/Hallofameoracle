
import React from 'react';
import { Trophy, Target, Award } from 'lucide-react'; // Removed Users, BarChart3 as they are not directly used here now
import PredictionForm from '@/components/PredictionForm'; // Import the new form

// Note: Card, CardContent, CardHeader, CardTitle are used by PredictionForm indirectly via ui components.
// We might not need direct imports here if PredictionForm handles its own card structure.
// For now, let's assume PredictionForm is self-contained or we adapt.

const Index: React.FC = () => {
  return (
    <div className="min-h-screen bg-gradient-to-br from-orange-50 via-white to-amber-50">
      {/* Header */}
      <div className="bg-gradient-to-r from-orange-600 to-amber-600 text-white py-12">
        <div className="container mx-auto px-4 text-center">
          <div className="flex items-center justify-center gap-3 mb-4">
            <Trophy className="h-12 w-12" />
            <Target className="h-10 w-10" />
            <Award className="h-12 w-12" />
          </div>
          <h1 className="text-5xl font-bold mb-4">NBA Hall of Fame Predictor (AI Model)</h1>
          <p className="text-xl opacity-90 max-w-2xl mx-auto">
            Enter player statistics and discover their likelihood of making it to the Basketball Hall of Fame using our AI model.
          </p>
        </div>
      </div>

      <div className="container mx-auto px-4 py-12">
        {/* 
          The old layout had a 2-column grid. 
          PredictionForm is designed to be a single column form with results displayed below it.
          We can either place it in one of the columns or let it take full width.
          For simplicity, let's center it.
        */}
        <div className="max-w-2xl mx-auto"> {/* Centering the form */}
          <PredictionForm />
        </div>
        
        {/* 
          The original page had a section for "Results" in the second column.
          PredictionForm.tsx now handles displaying its own results (probability and errors) directly within itself.
          So, we don't need a separate results card here unless we want to add other summary info.
          For now, PredictionForm is self-contained.
        */}
      </div>

      {/* You could add a footer or other sections here if needed */}
    </div>
  );
};

export default Index;
