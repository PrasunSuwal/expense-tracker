import { API_PATHS } from "./apiPaths";
import axiosInstance from "./axiosInstance";

const uploadImage = async (imageFile) => {
  const formData = new FormData();
  //APPEND IMAGE FILE TO FORM DATA
  formData.append("image", imageFile);

  try {
    const response = await axiosInstance.post(
      API_PATHS.IMAGE.UPLOAD_IMAGE,
      formData,
      {
        headers: {
          "Content-Type": "multipart/form-data", // set header for file upload
        },
      }
    );
    return response.data;
  } catch (error) {
    console.error("Error uplaoding the image", error);
    throw error; // rethrow error for handling
  }
};

export default uploadImage;
