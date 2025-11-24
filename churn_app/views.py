from django.shortcuts import render
import joblib
import numpy as np
import os
from django.conf import settings

# --- Paths ---
model_path = os.path.join(settings.BASE_DIR, 'churn_app', 'models', 'churn_Prediction_model_improved.pkl')
le_path = os.path.join(settings.BASE_DIR, 'churn_app', 'models', 'le_gender.pkl')

# --- Load once at startup ---
model = joblib.load(model_path)
le_gender = joblib.load(le_path)

# --- Threshold for business decision ---
CHURN_THRESHOLD = 0.4
def landing_page(request):
    return render(request, 'landing.html')

def predict_churn(request):
    if request.method == "POST":
        try:
            # --- Extract Inputs ---
            credit_score = float(request.POST['CreditScore'])
            age = float(request.POST['Age'])
            tenure = float(request.POST['Tenure'])
            balance = float(request.POST['Balance'])
            num_products = int(request.POST['NumOfProducts'])
            estimated_salary = float(request.POST['EstimatedSalary'])
            has_cr_card = int(request.POST['HasCrCard'])
            is_active_member = int(request.POST['IsActiveMember'])
            gender = request.POST['Gender'].strip().capitalize()

            # Validate gender input
            if gender not in le_gender.classes_:
                return render(request, "result.html", {
                    "result_text": f"❌ Invalid gender input: {gender}. Use Male or Female.",
                    "prob": None,
                    "prob_percent": 0,
                    "debug": {},
                    "is_positive": False
                })

            # --- Feature Engineering ---
            balance_salary_ratio = balance / (estimated_salary + 1e-6)
            tenure_by_age = tenure / (age + 1e-6)
            credit_score_given_age = credit_score / (age + 1e-6)
            gender_encoded = int(le_gender.transform([gender])[0])

            features = np.array([[  
                credit_score, age, tenure, balance, num_products,
                estimated_salary, balance_salary_ratio, tenure_by_age,
                credit_score_given_age, has_cr_card, is_active_member, gender_encoded
            ]], dtype=float)

            # --- Model Prediction ---
            probas = model.predict_proba(features)[0]
            classes = list(model.classes_)
            idx_for_churn = classes.index(1) if 1 in classes else -1
            churn_prob = float(probas[idx_for_churn])
            prob_percent = int(churn_prob * 100)
            raw_pred = int(model.predict(features)[0])
            decision = 1 if churn_prob >= CHURN_THRESHOLD else 0
            is_positive = decision == 1

            result_text = "⚠️ This customer IS LIKELY to churn." if is_positive else "✅ This customer is NOT likely to churn."

            # --- Rule-Based Explanation Engine ---
            reasons = []
            advice = []

            is_churn = is_positive  # True if likely to churn

            # CREDIT SCORE
            if credit_score < 580:
                reasons.append("Customer has a very low credit score, indicating high credit risk or previous payment issues.")
                advice.append("⚠️ Provide financial counseling or a secured credit card with low limits to rebuild trust.")
            else:
                if not is_churn:
                    reasons.append("Credit score is healthy and indicates responsible financial behavior.")
                    advice.append("✅ Credit score is healthy. Encourage continued responsible financial behavior.")

            # TENURE
            if tenure < 2:
                reasons.append("Customer has been with the bank for a short duration, indicating low brand loyalty.")
                advice.append("⚠️ Increase early engagement through personalized onboarding, rewards, and follow-ups.")
            else:
                if not is_churn:
                    reasons.append("Customer has some loyalty with the bank.")
                    advice.append("✅ Customer has some loyalty; maintain engagement with small perks.")

            # BALANCE
            if balance < 25000:
                reasons.append("Low account balance suggests limited financial engagement.")
                advice.append("⚠️ Introduce auto-savings plans or low-risk investment options.")
            elif balance > 200000:
                reasons.append("Very high balance may indicate underutilized funds or customer exploring alternatives.")
                advice.append("⚠️ Offer personalized investment advice or wealth management services.")
            else:
                if not is_churn:
                    reasons.append("Balance level is healthy and indicates moderate engagement.")
                    advice.append("✅ Balance level is healthy; maintain satisfaction with offers.")

            # NUMBER OF PRODUCTS
            if num_products <= 1:
                reasons.append("Customer holds only one banking product, increasing churn risk.")
                advice.append("⚠️ Cross-sell relevant products like insurance, Loans, or digital wallets, etc.")
            else:
                if not is_churn:
                    reasons.append("Customer has a moderate product portfolio.")
                    advice.append("✅ Customer has a moderate product portfolio; consider cross-selling strategically.")

            # ACCOUNT ACTIVITY
            if is_active_member == 0:
                reasons.append("Customer has low or no account activity recently.")
                advice.append("⚠️ Re-engage through personalized offers, app notifications, or loyalty-based campaigns.")
            else:
                if not is_churn:
                    reasons.append("Active account holder; engagement is good.")
                    advice.append("✅ Active account holder; continue providing seamless digital experiences.")

            # CREDIT CARD OWNERSHIP
            if has_cr_card == 0:
                reasons.append("Customer does not own a credit card — potentially less tied to the bank.")
                advice.append("⚠️ Promote an entry-level credit card with cashback or reward programs.")
            else:
                if not is_churn:
                    reasons.append("Customer holds a credit card, indicating stronger bank relationship.")
                    advice.append("✅ Credit card holder; offer bonus rewards or cashback upgrades.")

            # AGE
            if age < 25:
                reasons.append("Younger customers tend to explore better digital experiences and offers.")
                advice.append("⚠️ Focus on mobile-first experiences, student offers, and gamified banking.")
                if not is_churn:
                    advice.append("✅ Age is young; engage with mobile-first and gamified offerings.")
            elif age > 60:
                reasons.append("Senior customers often value stability and personalized service.")
                advice.append("⚠️ Offer easy-access customer support and senior benefits plans.")
                if not is_churn:
                    advice.append("✅ Senior age group; maintain personalized service and trust.")
            else:
                if not is_churn:
                    reasons.append("Age group is stable and indicates low churn risk.")
                    advice.append("✅ Age group is stable; maintain engagement with personalized offers.")

            # BALANCE-SALARY RATIO
            if balance_salary_ratio > 2:
                reasons.append("High balance-to-salary ratio may indicate unoptimized idle funds.")
                advice.append("⚠️ Offer investment or savings plans to improve utilization.")

            # If no reasons, default
            if not reasons:
                reasons.append("Customer profile appears balanced and stable with minimal churn risk factors.")
                advice.append("✅ Maintain strong engagement via continuous satisfaction monitoring and loyalty rewards.")

            debug = {
                "model_classes": [int(c) for c in classes],
                "raw_predict": int(raw_pred),
                "prob_index_used": int(idx_for_churn),
            }

            return render(request, "result.html", {
                "result_text": result_text,
                "prob": f"{churn_prob:.2f}",
                "prob_percent": prob_percent,
                "threshold": CHURN_THRESHOLD,
                "debug": debug,
                "is_positive": is_positive,
                "reasons": reasons,
                "advice": advice
            })

        except Exception as e:
            return render(request, "result.html", {
                "result_text": f"❌ Error processing input: {str(e)}",
                "prob": None,
                "prob_percent": 0,
                "debug": {},
                "is_positive": False
            })

    return render(request, "index.html")
