# ADR-007 · openFDA table fidelity spike

Ran before committing ~22 h to a custom SPL parser.
**Decision:** openFDA does not reliably preserve table structure. ADR-007 CONFIRMED: parse raw SPL XML with lxml. Challenge #3 would be unsolvable from openFDA JSON alone.


## vancomycin

- field type: `list`
- list of plain strings: `True`
- HTML table tags present: `False`
- pipe/tab structure: `False`
- **verdict: TABLES LIKELY FLATTENED TO TEXT**

```
DOSAGE AND ADMINISTRATION Infusion-related events are related to both the concentration and the rate of administration of vancomycin. Concentrations of no more than 5 mg/mL and rates of no more than 10 mg/min, are recommended in adults (see also age-specific recommendations). In selected patients in need of fluid restriction, a concentration up to 10 mg/mL may be used; use of such higher concentrations may increase the risk of infusion-related events. An infusion rate of 10 mg/min or less is associated with fewer infusion-related events (see ADVERSE REACTIONS ). Infusion-related events may occ
```

## levofloxacin

- field type: `list`
- list of plain strings: `True`
- HTML table tags present: `False`
- pipe/tab structure: `False`
- **verdict: TABLES LIKELY FLATTENED TO TEXT**

```
2 DOSAGE & ADMINISTRATION • Administer Levofloxacin Tablets to pediatric patients weighing 30 kg and greater only (2.1, 2.2). • Levofloxacin Tablets cannot be administered to pediatric patients who weigh less than 30 kg because of the limitations of the available strengths. Alternative formulations of levofloxacin may be considered for pediatric patients who weigh less than 30 kg (2.2). Dosage in Adult and Pediatric Patients with Creatinine Clearance greater than or equal to 50 mL/minute (2.1. 2.2) Type of Infection Dose Every 24 hours Duration (days) Nosocomial Pneumonia (1.1) 750 mg 7 to 14 
```

## enoxaparin

- field type: `list`
- list of plain strings: `True`
- HTML table tags present: `False`
- pipe/tab structure: `False`
- **verdict: TABLES LIKELY FLATTENED TO TEXT**

```
2 DOSAGE AND ADMINISTRATION See full prescribing information for dosing and administration information. ( 2 ) 2.1 Pretreatment Evaluation Evaluate all patients for a bleeding disorder before starting enoxaparin sodium treatment, unless treatment is urgently needed. 2.2 Adult Dosage Abdominal Surgery The recommended dose of enoxaparin sodium is 40 mg by subcutaneous injection once a day (with the initial dose given 2 hours prior to surgery) in patients undergoing abdominal surgery who are at risk for thromboembolic complications. The usual duration of administration is 7 to 10 days [see Clinica
```

## rivaroxaban

- field type: `list`
- list of plain strings: `True`
- HTML table tags present: `False`
- pipe/tab structure: `False`
- **verdict: TABLES LIKELY FLATTENED TO TEXT**

```
2 DOSAGE AND ADMINISTRATION Nonvalvular Atrial Fibrillation : 15 or 20 mg, once daily with food ( 2.1 ) Treatment of DVT and/or PE : 15 mg orally twice daily with food for the first 21 days followed by 20 mg orally once daily with food for the remaining treatment ( 2.1 ) Reduction in the Risk of Recurrence of DVT and/or PE in patients at continued risk for DVT and/or PE : 10 mg once daily with or without food, after at least 6 months of standard anticoagulant treatment ( 2.1 ) Prophylaxis of DVT Following Hip or Knee Replacement Surgery : 10 mg orally once daily with or without food ( 2.1 ) Pr
```

## dabigatran

- field type: `list`
- list of plain strings: `True`
- HTML table tags present: `False`
- pipe/tab structure: `False`
- **verdict: TABLES LIKELY FLATTENED TO TEXT**

```
2 DOSAGE AND ADMINISTRATION • Non-valvular Atrial Fibrillation in Adult Patients: o For patients with CrCl >30 mL/min: 150 mg orally, twice daily ( 2.2 ) o For patients with CrCl 15 to 30 mL/min: 75 mg orally, twice daily ( 2.2 ) • Treatment of DVT and PE in Adult Patients : o For patients with CrCl >30 mL/min: 150 mg orally, twice daily after 5 to 10 days of parenteral anticoagulation ( 2.2 ) • Reduction in the Risk of Recurrence of DVT and PE in Adult Patients : o For patients with CrCl >30 mL/min: 150 mg orally, twice daily after previous treatment ( 2.2 ) • Prophylaxis of DVT and PE Follow
```
