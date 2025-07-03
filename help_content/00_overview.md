# Overview of Code Structure
The application's Python script is broadly divided into two main parts:

1.  **Customizable Section:** This is where you'll make most of your modifications to adapt the application to new sensors, data processing logic, or GUI layouts. It's found near the top of the script, demarcated by comments like `# Start of customizable section` and `# End of customizable section`.
2.  **Backend/Core Logic:** This section contains the underlying framework for BLE communication, GUI management, data buffering, and other core functionalities. While powerful, changes here require a deeper understanding of the system and are generally not needed for typical customizations. It's usually marked with a "*don’t-touch-unless-you-know-what-you’re-doing*" type of warning.

The customization process usually involves interacting with specific, well-defined parts of the customizable section.