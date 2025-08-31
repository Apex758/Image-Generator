// Define types for our lesson plan data
export interface LessonPlan {
  id: string;
  title: string;
  gradeLevel: string;
  subject: string;
  content: string;
}

// Sample lesson plans data
export const lessonPlans: LessonPlan[] = [
  {
    id: "lp-001",
    title: "Introduction to Fractions",
    gradeLevel: "Elementary",
    subject: "Mathematics",
    content: "Objective: Students will understand the concept of fractions as parts of a whole.\n\nMaterials: Fraction circles, worksheets, interactive whiteboard.\n\nActivities:\n1. Begin by showing students a whole pizza and then cutting it into equal parts.\n2. Introduce the terms numerator and denominator.\n3. Have students practice identifying fractions using visual models.\n4. Group activity: Students create their own fraction models using paper plates.\n\nAssessment: Students will complete a worksheet identifying fractions in various models."
  },
  {
    id: "lp-002",
    title: "The Water Cycle",
    gradeLevel: "Elementary",
    subject: "Science",
    content: "Objective: Students will be able to describe the stages of the water cycle.\n\nMaterials: Water cycle diagram, clear containers, ice, hot plate (teacher use only).\n\nActivities:\n1. Introduce the water cycle with an illustrated diagram.\n2. Demonstrate evaporation by heating water (teacher demonstration).\n3. Show condensation using ice on the outside of a container.\n4. Students create their own water cycle diagrams.\n\nAssessment: Students will label the stages of the water cycle and explain each process."
  },
  {
    id: "lp-003",
    title: "Introduction to Shakespeare",
    gradeLevel: "High School",
    subject: "English",
    content: "Objective: Students will understand the historical context of Shakespeare's works and analyze key themes.\n\nMaterials: Copies of Romeo and Juliet, video clips of performances, handouts on Elizabethan England.\n\nActivities:\n1. Present background information on Shakespeare and the Elizabethan era.\n2. Read and analyze key scenes from Romeo and Juliet.\n3. Compare original text with modern adaptations.\n4. Group discussion on universal themes in Shakespeare's works.\n\nAssessment: Students will write a short essay analyzing a theme from Romeo and Juliet."
  },
  {
    id: "lp-004",
    title: "Chemical Reactions",
    gradeLevel: "Middle School",
    subject: "Science",
    content: "Objective: Students will identify evidence of chemical reactions and understand the conservation of mass.\n\nMaterials: Baking soda, vinegar, balloons, scales, safety goggles.\n\nActivities:\n1. Safety review for handling materials.\n2. Demonstrate baking soda and vinegar reaction.\n3. Students conduct experiments measuring mass before and after reactions.\n4. Document observations and results in lab notebooks.\n\nAssessment: Lab report analyzing the evidence of chemical reactions and explaining conservation of mass."
  },
  {
    id: "lp-005",
    title: "Ancient Egypt Civilization",
    gradeLevel: "Middle School",
    subject: "History",
    content: "Objective: Students will understand key aspects of Ancient Egyptian civilization including social structure, religion, and achievements.\n\nMaterials: Maps, images of artifacts, hieroglyphic chart, video on pyramid construction.\n\nActivities:\n1. Locate Egypt on a map and discuss the importance of the Nile River.\n2. Examine the social pyramid of Ancient Egyptian society.\n3. Explore religious beliefs and practices.\n4. Students create a travel brochure for Ancient Egypt highlighting key achievements.\n\nAssessment: Students will create a museum exhibit display on one aspect of Ancient Egyptian civilization."
  },
  {
    id: "lp-006",
    title: "Photosynthesis",
    gradeLevel: "High School",
    subject: "Biology",
    content: "Objective: Students will understand the process of photosynthesis and its importance to life on Earth.\n\nMaterials: Plant specimens, microscopes, diagrams, colored pencils, lab equipment for chlorophyll extraction.\n\nActivities:\n1. Review the chemical equation for photosynthesis.\n2. Examine leaf structures under microscopes.\n3. Conduct an experiment measuring oxygen production in water plants under different light conditions.\n4. Create detailed diagrams of the light-dependent and light-independent reactions.\n\nAssessment: Lab report and quiz on the stages of photosynthesis."
  },
  {
    id: "lp-007",
    title: "Introduction to Coding",
    gradeLevel: "Elementary",
    subject: "Computer Science",
    content: "Objective: Students will understand basic programming concepts and create simple algorithms.\n\nMaterials: Computers with block-based programming software, unplugged activity cards.\n\nActivities:\n1. Introduce computational thinking through unplugged activities.\n2. Demonstrate basic block-based programming concepts.\n3. Students work in pairs to solve simple programming challenges.\n4. Share and explain solutions with the class.\n\nAssessment: Students will create a simple animated story using block-based programming."
  },
  {
    id: "lp-008",
    title: "World War II: Causes and Impact",
    gradeLevel: "High School",
    subject: "History",
    content: "Objective: Students will analyze the causes, major events, and global impact of World War II.\n\nMaterials: Maps, primary source documents, video testimonials, timeline materials.\n\nActivities:\n1. Analyze the Treaty of Versailles and its impact on Germany after WWI.\n2. Examine the rise of totalitarian regimes in the 1930s.\n3. Create a collaborative timeline of major WWII events.\n4. Discuss the Holocaust and human rights implications.\n5. Analyze the post-war world order and the beginning of the Cold War.\n\nAssessment: Research paper on a specific aspect of WWII and its long-term impact."
  },
  {
    id: "lp-009",
    title: "Poetry Analysis and Creation",
    gradeLevel: "Middle School",
    subject: "English",
    content: "Objective: Students will analyze poetic devices and create original poems using these techniques.\n\nMaterials: Poetry examples, literary device handouts, writing materials.\n\nActivities:\n1. Introduce and identify various poetic devices (metaphor, simile, alliteration, etc.).\n2. Analyze poems in small groups, identifying devices and discussing their effect.\n3. Model the poetry writing process.\n4. Students create original poems using at least three poetic devices.\n\nAssessment: Original poem and written analysis explaining the devices used and their intended effect."
  },
  {
    id: "lp-010",
    title: "Ecosystems and Biodiversity",
    gradeLevel: "Middle School",
    subject: "Science",
    content: "Objective: Students will understand ecosystem components, interactions, and the importance of biodiversity.\n\nMaterials: Ecosystem diagrams, food web cards, local ecosystem case study materials.\n\nActivities:\n1. Define and discuss ecosystem components (biotic and abiotic factors).\n2. Create food webs using cards and string to show interconnections.\n3. Analyze a local ecosystem and identify biodiversity factors.\n4. Discuss human impacts on ecosystems and conservation strategies.\n\nAssessment: Students will create a model of a specific ecosystem, explaining interactions and biodiversity importance."
  }
];

// Helper function to get unique grade levels from the data
export const getUniqueGradeLevels = (): string[] => {
  const gradeLevels = lessonPlans.map(plan => plan.gradeLevel);
  return [...new Set(gradeLevels)];
};

// Helper function to get unique subjects from the data
export const getUniqueSubjects = (): string[] => {
  const subjects = lessonPlans.map(plan => plan.subject);
  return [...new Set(subjects)];
};