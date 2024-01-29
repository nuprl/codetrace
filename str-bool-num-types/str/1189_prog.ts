export default function parseJob({ skills, seniorityLevels }) {
  return ({
    id,
    title,
    description,
    seniorities: seniorityIds,
    mainSkills: mainSkillIds,
    completeSkills: completeSkillIds,
    location,
    remote,
    startingDate,
  }: Job) => {
    const seniorities = findAll(seniorityLevels, seniorityIds) as Seniority[];
    const mainSkills = findAll(skills, mainSkillIds) as Skill[];
    const completeSkills = findAll(skills, completeSkillIds) as Skill[];
    return {
      id,
      title,
      description,
      location,
      remote,
      startingDate,
      experience: getExperienceRange(seniorities),
      salary: getSalaryRange(seniorities),
      mainSkills: getSkills(mainSkills),
      completeSkills: getSkills(completeSkills),
    };
  };
}

function find(items: { id: string }[], id: string) {
  return items.find((item) => item.id === id);
}

function findAll(items: { id: string }[], ids: string[] = []) {
  return ids.map((id) => find(items, id));
}

function getSkills(skills: Skill[]) {
  return skills.map((skill) => skill.name);
}

function getExperienceRange(seniorities: Seniority[]) {
  return seniorities
    .reduce(
      (acc, seniority) => [
        Math.min(seniority.experience.min, acc[0]),
        Math.max(seniority.experience.max || seniority.experience.min, acc[1]),
      ],
      [Infinity, 0]
    )
    .join(" - ");
}

function getSalaryRange(seniorities: Seniority[]) {
  return seniorities
    .reduce(
      (acc, seniority) => [
        Math.round(Math.min(seniority.salary.base, acc[0])),
        Math.round(Math.max(seniority.salary.base, acc[1])),
      ],
      [Infinity, 0]
    )
    .join(" - ");
}

type Skill = { id: string; name: <FILL> };

type Seniority = {
  id: string;
  experience: { max: number; min: number };
  salary: { base: number };
};

export type Job = {
  id: string;
  title: string;
  description: string;
  seniorities: string[];
  mainSkills: string[];
  completeSkills: string[];
  location: string;
  remote: boolean;
  startingDate: number;
};
