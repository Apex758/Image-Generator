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
    gradeLevel: "3",
    subject: "Math",
    content: "Objective: Students will understand the concept of fractions as parts of a whole.\n\nMaterials: Fraction circles, worksheets, interactive whiteboard.\n\nActivities:\n1. Begin by showing students a whole pizza and then cutting it into equal parts.\n2. Introduce the terms numerator and denominator.\n3. Have students practice identifying fractions using visual models.\n4. Group activity: Students create their own fraction models using paper plates.\n\nAssessment: Students will complete a worksheet identifying fractions in various models."
  },
  {
    id: "lp-002",
    title: "The Water Cycle",
    gradeLevel: "2",
    subject: "Science",
    content: "Objective: Students will be able to describe the stages of the water cycle.\n\nMaterials: Water cycle diagram, clear containers, ice, hot plate (teacher use only).\n\nActivities:\n1. Introduce the water cycle with an illustrated diagram.\n2. Demonstrate evaporation by heating water (teacher demonstration).\n3. Show condensation using ice on the outside of a container.\n4. Students create their own water cycle diagrams.\n\nAssessment: Students will label the stages of the water cycle and explain each process."
  },
  {
    id: "lp-003",
    title: "Story Writing and Reading Comprehension",
    gradeLevel: "1",
    subject: "Language Arts",
    content: "Objective: Students will develop reading comprehension skills and write simple stories.\n\nMaterials: Picture books, writing paper, crayons, story templates.\n\nActivities:\n1. Read a simple picture book aloud to the class.\n2. Discuss the main characters, setting, and plot.\n3. Students draw pictures to retell the story sequence.\n4. Guide students in writing their own simple stories with pictures.\n\nAssessment: Students will share their stories and answer questions about the book we read together."
  },
  {
    id: "lp-004",
    title: "Simple Machines in Our World",
    gradeLevel: "6",
    subject: "Science",
    content: "Objective: Students will identify and understand how simple machines make work easier.\n\nMaterials: Examples of simple machines, diagrams, building materials for models.\n\nActivities:\n1. Introduce the six types of simple machines with real examples.\n2. Students work in groups to identify simple machines around the classroom.\n3. Build simple models using everyday materials (levers, pulleys, etc.).\n4. Test and demonstrate how each machine makes work easier.\n\nAssessment: Students will create a poster showing examples of simple machines in their daily lives."
  },
  {
    id: "lp-005",
    title: "Community Helpers and Local Government",
    gradeLevel: "K",
    subject: "Social Studies",
    content: "Objective: Students will understand the roles of community helpers and basic local government.\n\nMaterials: Pictures of community helpers, role-play costumes, maps of local community.\n\nActivities:\n1. Show pictures and discuss different community helpers (police, firefighters, teachers, etc.).\n2. Role-play being different community helpers.\n3. Take a simple walking tour around the school to see community helpers at work.\n4. Create a class book about 'Our Community Helpers'.\n\nAssessment: Students will draw and tell about their favorite community helper and what they do."
  },
  {
    id: "lp-006",
    title: "Addition and Subtraction Word Problems",
    gradeLevel: "2",
    subject: "Math",
    content: "Objective: Students will solve addition and subtraction word problems using strategies and manipulatives.\n\nMaterials: Counting bears, number lines, word problem cards, math journals.\n\nActivities:\n1. Read word problems aloud and identify key information.\n2. Use counting bears to model addition and subtraction problems.\n3. Practice drawing pictures to represent word problems.\n4. Students create their own word problems for classmates to solve.\n\nAssessment: Students will solve a variety of word problems and explain their thinking process."
  },
  {
    id: "lp-007",
    title: "Animal Habitats and Adaptations",
    gradeLevel: "1",
    subject: "Science",
    content: "Objective: Students will understand how animals are adapted to live in different habitats.\n\nMaterials: Animal pictures, habitat cards, craft materials for dioramas.\n\nActivities:\n1. Discuss different habitats (forest, desert, ocean, etc.).\n2. Match animals to their appropriate habitats.\n3. Explore how animal features help them survive in their habitats.\n4. Create simple habitat dioramas in shoeboxes.\n\nAssessment: Students will present their dioramas and explain how their chosen animal is adapted to its habitat."
  },
  {
    id: "lp-008",
    title: "Maps, Globes, and Directions",
    gradeLevel: "4",
    subject: "Social Studies",
    content: "Objective: Students will understand how to read maps and use cardinal directions.\n\nMaterials: World map, globes, compass, local area maps, map symbols chart.\n\nActivities:\n1. Introduce cardinal directions (North, South, East, West) using a compass.\n2. Practice finding locations on a globe and world map.\n3. Learn common map symbols and create a legend.\n4. Draw a simple map of the classroom or school playground.\n\nAssessment: Students will use a map to give directions from one location to another using cardinal directions."
  },
  {
    id: "lp-009",
    title: "Reading Fluency and Expression",
    gradeLevel: "4",
    subject: "Language Arts",
    content: "Objective: Students will improve reading fluency and learn to read with appropriate expression.\n\nMaterials: Level-appropriate books, recording devices, expression cards, reading passages.\n\nActivities:\n1. Model fluent reading with appropriate expression and pacing.\n2. Practice choral reading with the whole class.\n3. Partner reading where students take turns reading aloud.\n4. Record students reading and play back for self-evaluation.\n\nAssessment: Students will read a passage aloud demonstrating improved fluency and expression."
  },
  {
    id: "lp-010",
    title: "Multiplication Facts and Arrays",
    gradeLevel: "5",
    subject: "Math",
    content: "Objective: Students will understand multiplication concepts using arrays and memorize basic multiplication facts.\n\nMaterials: Manipulatives for arrays, multiplication charts, fact cards, grid paper.\n\nActivities:\n1. Use physical objects to create arrays and understand multiplication as repeated addition.\n2. Practice skip counting to reinforce multiplication patterns.\n3. Play multiplication games and use fact families.\n4. Create visual arrays on grid paper for different multiplication problems.\n\nAssessment: Students will demonstrate fluency with multiplication facts through timed practice and array creation."
  },
  {
    id: "lp-011",
    title: "Weather Patterns and Seasons",
    gradeLevel: "K",
    subject: "Science",
    content: "Objective: Students will observe and describe weather patterns and seasonal changes.\n\nMaterials: Weather chart, thermometer, weather symbols, seasonal clothing items.\n\nActivities:\n1. Daily weather observations and charting.\n2. Discuss appropriate clothing for different weather conditions.\n3. Explore the four seasons and their characteristics.\n4. Create a class weather book with drawings and simple descriptions.\n\nAssessment: Students will identify weather patterns and describe seasonal changes through drawings and oral explanations."
  },
  {
    id: "lp-012",
    title: "American Symbols and Holidays",
    gradeLevel: "3",
    subject: "Social Studies",
    content: "Objective: Students will identify American symbols and understand the significance of national holidays.\n\nMaterials: Pictures of American symbols, flag, maps, holiday timeline, craft materials.\n\nActivities:\n1. Learn about American symbols (flag, eagle, Statue of Liberty, etc.) and their meanings.\n2. Discuss major American holidays and why we celebrate them.\n3. Create a timeline of national holidays throughout the year.\n4. Design and create their own American symbol with explanation.\n\nAssessment: Students will explain the significance of at least three American symbols and two national holidays."
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