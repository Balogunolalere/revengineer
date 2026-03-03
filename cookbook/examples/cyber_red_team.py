#!/usr/bin/env python3
"""
Example: Adversarial Cyber Red Team — the monster build.

A 10-agent DAG that runs a full offensive + defensive security
analysis pipeline. Agents attack, defend, and audit each other's
work in waves, producing a comprehensive security assessment.

Architecture (DAG):

  ┌─────────────────┐  ┌──────────────────┐  ┌─────────────────┐
  │ Recon Specialist │  │ Vuln Researcher  │  │ Threat Intel    │
  │   (OSINT + ASM)  │  │  (CVE + 0-day)   │  │   Analyst       │
  └────────┬────────┘  └────────┬─────────┘  └────────┬────────┘
           │                    │                      │
           └────────┬───────────┘                      │
                    │                                  │
           ┌────────▼─────────┐               ┌───────▼────────┐
           │ Exploit Developer│               │ Malware Analyst │
           │ (builds attacks) │               │ (threat models) │
           └────────┬─────────┘               └───────┬────────┘
                    │                                  │
           ┌────────▼─────────┐                        │
           │ Defense Architect│◄───────────────────────┘
           │ (countermeasures)│
           └────────┬─────────┘
                    │
        ┌───────────┼───────────┐
        │           │           │
 ┌──────▼─────┐ ┌──▼────────┐ ┌▼─────────────┐
 │ Purple Team│ │ Compliance │ │ Incident     │
 │  Validator │ │   Auditor  │ │ Response Plan│
 └──────┬─────┘ └──┬────────┘ └┬─────────────┘
        │           │           │
        └───────────┼───────────┘
                    │
           ┌────────▼─────────┐
           │   CISO Briefing  │
           │  (final report)  │
           └──────────────────┘

Prerequisites:
    1. Start your DeepSeek API proxy:  uv run python deepseek_api.py
    2. Run this script:                uv run python cookbook/examples/cyber_red_team.py

    Optional — pass a target description:
        uv run python cookbook/examples/cyber_red_team.py "a fintech startup running
        Kubernetes on AWS with a React frontend, Python microservices, PostgreSQL,
        Redis, and a mobile app"
"""

import asyncio
import sys

from cookbook.swarm import (
    Swarm, SwarmConfig, SwarmMode,
    SwarmPlan, AgentSpec,
)


# Default target if none provided
DEFAULT_TARGET = (
    "A mid-size SaaS company (Series B, 50 engineers) running: "
    "- Kubernetes on AWS (EKS) with Terraform-managed infrastructure\n"
    "- React/Next.js frontend served via CloudFront\n"
    "- Python (FastAPI) and Go microservices behind an API gateway\n"
    "- PostgreSQL (RDS) + Redis (ElastiCache) + S3 for storage\n"
    "- GitHub Actions CI/CD with Argo CD for deployments\n"
    "- OAuth2/OIDC authentication via Auth0\n"
    "- Mobile apps (React Native) on iOS and Android\n"
    "- Internal admin dashboard with role-based access\n"
    "- Third-party integrations: Stripe, Twilio, SendGrid, Datadog"
)


async def main():
    target = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else DEFAULT_TARGET

    config = SwarmConfig(
        api_base="http://localhost:8000/v1",
        max_parallel=5,
        agent_timeout=180,  # security analysis needs more time (now actually works)
        max_retries=3,      # more retries for flaky upstream
        max_tokens=4096,
        output_dir=".",
    )

    # ── Wave 1: Reconnaissance (parallel) ────────────────────────

    recon = AgentSpec(
        role="Recon Specialist",
        task=(
            f"Perform a comprehensive attack surface mapping for the following target:\n\n"
            f"{target}\n\n"
            "Enumerate:\n"
            "1. All externally exposed services and their likely ports/protocols\n"
            "2. DNS configuration attack vectors (subdomain takeover, zone transfer)\n"
            "3. Cloud metadata service exposure risks\n"
            "4. API endpoint discovery methodology\n"
            "5. Third-party integration attack surfaces\n"
            "6. Mobile app reverse engineering opportunities\n"
            "7. OSINT data leakage (GitHub, Docker Hub, npm, job postings)\n\n"
            "For each finding, rate the exposure level (Critical/High/Medium/Low) "
            "and describe what an attacker would do with it."
        ),
        tools=["search"],
        system_prompt=(
            "You are an elite OSINT and attack surface management specialist. "
            "You think like a nation-state APT group doing initial reconnaissance. "
            "Be thorough, creative, and ruthlessly practical."
        ),
        priority=5,
    )

    vuln_researcher = AgentSpec(
        role="Vulnerability Researcher",
        task=(
            f"Research known and potential vulnerabilities in this technology stack:\n\n"
            f"{target}\n\n"
            "For each component (Kubernetes/EKS, FastAPI, Go services, PostgreSQL, "
            "Redis, Auth0, React Native, Terraform, Argo CD, GitHub Actions):\n"
            "1. List the most critical CVEs from 2024-2026\n"
            "2. Identify common misconfigurations that lead to compromise\n"
            "3. Note any 0-day classes these components are historically prone to\n"
            "4. Assess supply chain attack vectors (dependency confusion, typosquatting)\n"
            "5. Evaluate container escape and privilege escalation paths\n\n"
            "Output a prioritized vulnerability matrix with CVSS scores where available."
        ),
        tools=["search"],
        system_prompt=(
            "You are a vulnerability researcher with deep knowledge of CVE databases, "
            "exploit-db, and security advisories. You track 0-days and have expertise "
            "in cloud-native and container security."
        ),
        priority=5,
    )

    threat_intel = AgentSpec(
        role="Threat Intelligence Analyst",
        task=(
            f"Produce a threat intelligence briefing for this target:\n\n"
            f"{target}\n\n"
            "Cover:\n"
            "1. APT groups known to target fintech/SaaS companies (TTPs, IOCs)\n"
            "2. Ransomware groups currently active against similar stacks\n"
            "3. Recent supply chain attacks relevant to this tech stack\n"
            "4. Insider threat scenarios specific to a 50-engineer team\n"
            "5. Current threat landscape for AWS-hosted Kubernetes workloads\n"
            "6. Social engineering attack vectors (spearphishing, credential stuffing)\n"
            "7. Map findings to MITRE ATT&CK techniques\n\n"
            "Include real-world breach examples from 2024-2026 as case studies."
        ),
        tools=["search"],
        system_prompt=(
            "You are a threat intelligence analyst at a top-tier ISAC. "
            "You track APT campaigns, ransomware operations, and supply chain attacks. "
            "You think in terms of MITRE ATT&CK and the Diamond Model."
        ),
        priority=4,
    )

    # ── Wave 2: Offense (depends on recon + vuln research) ────────

    exploit_dev = AgentSpec(
        role="Exploit Developer",
        task=(
            "Based on the Recon Specialist's attack surface mapping and the "
            "Vulnerability Researcher's findings, develop a comprehensive "
            "attack playbook:\n\n"
            "For the TOP 10 most promising attack chains:\n"
            "1. Describe the full kill chain (initial access → lateral movement → "
            "   privilege escalation → objective)\n"
            "2. Provide specific techniques at each stage\n"
            "3. Estimate likelihood of success and required skill level\n"
            "4. Note what defensive telemetry each step would generate\n"
            "5. Identify the 'crown jewels' each chain targets\n\n"
            "Also design 3 NOVEL attack chains that combine multiple low-severity "
            "findings into critical exploit paths (chained attacks).\n\n"
            "Rate each chain: effort (days), noise level (1-10), impact (1-10)."
        ),
        depends_on=[recon.agent_id, vuln_researcher.agent_id],
        system_prompt=(
            "You are a red team operator and exploit developer. You chain "
            "vulnerabilities creatively, think in kill chains, and design attacks "
            "that would bypass typical enterprise defenses. You've done hundreds "
            "of penetration tests."
        ),
        priority=5,
        timeout=300,  # heavy task with large dependent context
    )

    malware_analyst = AgentSpec(
        role="Malware & Threat Modeling Analyst",
        task=(
            "Based on the Threat Intelligence Analyst's briefing, build detailed "
            "threat models for the target:\n\n"
            "1. Create STRIDE threat models for each major component\n"
            "2. Map the most likely attack scenarios to the kill chain\n"
            "3. Model data flow diagrams showing trust boundaries\n"
            "4. Identify the most valuable data assets and their exposure\n"
            "5. Assess the blast radius of each compromise scenario\n"
            "6. Model the attacker's ROI — which attacks give max payoff "
            "   for minimum effort?\n\n"
            "Produce an attack tree for the 3 most critical assets."
        ),
        depends_on=[threat_intel.agent_id],
        system_prompt=(
            "You are a threat modeling expert who uses STRIDE, attack trees, "
            "and data flow analysis. You think systematically about trust "
            "boundaries and blast radius. You've modeled threats for Fortune 500s."
        ),
        priority=4,
    )

    # ── Wave 3: Defense (depends on offense) ──────────────────────

    defense_architect = AgentSpec(
        role="Defense Architect",
        task=(
            "You have the full offensive output: attack surface, vulnerabilities, "
            "exploit chains, threat intelligence, and threat models.\n\n"
            "Design a comprehensive defense architecture:\n\n"
            "1. **Zero Trust Architecture**: Network segmentation, micro-segmentation, "
            "   identity-based access controls — specific to this K8s/AWS stack\n"
            "2. **Detection Engineering**: Write detection rules (Sigma/Yara format) "
            "   for the top 10 attack chains identified\n"
            "3. **Hardening Playbook**: Specific hardening steps for each component "
            "   (K8s, RDS, Redis, Auth0, CI/CD, container images)\n"
            "4. **Security Architecture Diagram**: Describe the ideal security "
            "   layers (WAF, API gateway, service mesh, secrets management, etc.)\n"
            "5. **Quick Wins**: Top 10 changes that can be implemented THIS WEEK "
            "   with maximum risk reduction\n\n"
            "For each recommendation: effort (hours), risk reduction (1-10), "
            "and which attack chains it neutralizes."
        ),
        depends_on=[exploit_dev.agent_id, malware_analyst.agent_id],
        system_prompt=(
            "You are a senior security architect who designs defense-in-depth "
            "strategies. You're practical — you know what engineering teams will "
            "actually implement. You think in terms of layers, blast radius "
            "reduction, and detection-as-code."
        ),
        priority=5,
        timeout=300,  # heavy task with large dependent context
    )

    # ── Wave 4: Validation (depends on defense, parallel) ─────────

    purple_team = AgentSpec(
        role="Purple Team Validator",
        task=(
            "Review the Exploit Developer's attack playbook AND the Defense "
            "Architect's countermeasures. Your job is adversarial validation:\n\n"
            "1. For each proposed defense, describe how a skilled attacker would "
            "   attempt to BYPASS it\n"
            "2. Identify defensive gaps — attacks that NO proposed defense covers\n"
            "3. Rate each defense's effectiveness: Would it actually stop the "
            "   attack in practice? (Yes/Partial/No + explanation)\n"
            "4. Propose 5 additional attack scenarios the red team MISSED\n"
            "5. Suggest detection improvements that would catch evasion attempts\n\n"
            "Be brutally honest. The goal is to find what's still broken."
        ),
        depends_on=[defense_architect.agent_id],
        system_prompt=(
            "You are a purple team lead who validates both offense and defense. "
            "You find holes in defenses and attacks that red teams miss. "
            "You're the most paranoid person in the room."
        ),
        priority=4,
    )

    compliance_auditor = AgentSpec(
        role="Compliance & Governance Auditor",
        task=(
            "Given the full security assessment (recon, vulns, attacks, defenses), "
            "perform a compliance gap analysis:\n\n"
            "1. **SOC 2 Type II**: Map findings to Trust Services Criteria — "
            "   which controls are missing or weak?\n"
            "2. **PCI DSS 4.0**: If the company handles payment data (Stripe), "
            "   assess compliance gaps\n"
            "3. **GDPR/CCPA**: Data protection impact assessment based on the "
            "   identified data flows and vulnerabilities\n"
            "4. **ISO 27001**: Gap analysis against Annex A controls\n"
            "5. **Regulatory Risk**: What would a regulator cite them for TODAY?\n\n"
            "Produce a remediation priority matrix: gap, regulation, risk level, "
            "estimated effort to close."
        ),
        depends_on=[defense_architect.agent_id],
        system_prompt=(
            "You are a GRC (Governance, Risk, Compliance) specialist with "
            "audit experience across SOC 2, PCI DSS, GDPR, and ISO 27001. "
            "You translate technical findings into compliance language."
        ),
        priority=3,
    )

    ir_planner = AgentSpec(
        role="Incident Response Planner",
        task=(
            "Based on the threat models and attack chains identified, build "
            "a complete incident response framework:\n\n"
            "1. **IR Playbooks**: Write step-by-step runbooks for the 5 most "
            "   likely incident scenarios (ransomware, data breach, supply chain "
            "   compromise, insider threat, API key leak)\n"
            "2. **Detection → Response Timeline**: For each scenario, define "
            "   detection triggers, triage steps, containment actions, and "
            "   recovery procedures with time targets\n"
            "3. **Communication Templates**: Draft notification templates for "
            "   customers, regulators, and the board\n"
            "4. **Tabletop Exercise**: Design a 2-hour tabletop exercise "
            "   scenario based on the most critical attack chain\n"
            "5. **Forensic Readiness**: What logging and evidence preservation "
            "   must be in place BEFORE an incident?\n\n"
            "Include specific tool recommendations (SIEM, EDR, forensic tools)."
        ),
        depends_on=[defense_architect.agent_id],
        system_prompt=(
            "You are a DFIR (Digital Forensics & Incident Response) team lead. "
            "You've handled breaches at scale and know what breaks during real "
            "incidents. You write runbooks that work at 3 AM."
        ),
        priority=3,
    )

    # ── Wave 5: Final briefing (depends on all validation) ────────

    ciso_briefing = AgentSpec(
        role="CISO Briefing Author",
        task=(
            "Produce an executive-level CISO security briefing that synthesizes "
            "ALL findings from the entire assessment:\n\n"
            "Structure:\n"
            "1. **Executive Summary** (1 page): Overall risk posture, top 3 risks, "
            "   overall risk rating (Critical/High/Medium/Low)\n"
            "2. **Risk Heat Map**: Matrix of likelihood vs impact for top 15 risks\n"
            "3. **Attack Narrative**: Tell the story of the #1 most likely breach "
            "   scenario — how it starts, what the attacker does, what they steal, "
            "   and what it costs the company\n"
            "4. **Remediation Roadmap**: 30/60/90-day plan with specific actions, "
            "   owners, and success metrics\n"
            "5. **Budget Ask**: Estimated security investment needed, broken down "
            "   by: tooling, headcount, training, third-party audits\n"
            "6. **Board-Ready Metrics**: 5 KPIs to track security posture\n"
            "7. **Bottom Line**: What happens if we do nothing?\n\n"
            "Tone: professional, urgent, data-driven. No jargon without explanation."
        ),
        depends_on=[
            purple_team.agent_id,
            compliance_auditor.agent_id,
            ir_planner.agent_id,
        ],
        system_prompt=(
            "You are a virtual CISO writing a board-level security briefing. "
            "You translate technical risk into business impact. You know how "
            "to make executives care about security without overwhelming them. "
            "Your recommendations are specific, costed, and prioritized."
        ),
        priority=5,
    )

    # ── Build the plan ────────────────────────────────────────────

    plan = SwarmPlan(
        goal=f"Full adversarial security assessment of: {target[:100]}...",
        agents=[
            # Wave 1 — parallel recon
            recon, vuln_researcher, threat_intel,
            # Wave 2 — offense
            exploit_dev, malware_analyst,
            # Wave 3 — defense
            defense_architect,
            # Wave 4 — validation (parallel)
            purple_team, compliance_auditor, ir_planner,
            # Wave 5 — executive briefing
            ciso_briefing,
        ],
        strategy=(
            "5-wave adversarial pipeline: "
            "Recon + Vuln Research + Threat Intel (parallel) → "
            "Exploit Development + Threat Modeling → "
            "Defense Architecture → "
            "Purple Team + Compliance + IR Planning (parallel) → "
            "CISO Executive Briefing"
        ),
    )

    # Validate the DAG
    errors = plan.validate()
    if errors:
        print(f"Plan validation errors: {errors}")
        return

    result = await Swarm(
        plan.goal,
        config=config,
        mode=SwarmMode.MANUAL,
        plan=plan,
        verbose=True,
    )

    print(f"\n{'='*60}")
    print(f"SECURITY ASSESSMENT COMPLETE")
    print(f"  {len(result.successful)} agents succeeded, {len(result.failed)} failed")
    print(f"  Duration: {result.duration:.1f}s")
    print(f"{'='*60}\n")
    print(result.synthesis)


if __name__ == "__main__":
    asyncio.run(main())
